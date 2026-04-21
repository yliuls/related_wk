Title: G-Adaptivity: optimised graph-based mesh relocation for finite element methods
Abstract: We present a novel, and effective, approach to achieve optimal mesh relocation in finite element methods (FEMs). The cost and accuracy of FEMs is critically dependent on the choice of mesh points. Mesh relocation (r-adaptivity) seeks to optimise the mesh geometry to obtain the best solution accuracy at given computational budget. Classical r-adaptivity relies on the solution of a separate nonlinear "meshing" PDE to determine mesh point locations. This incurs significant cost at remeshing, and relies on estimates that relate interpolation-and FEM-error. Recent machine learning approaches have focused on the construction of fast surrogates for such classical methods. Instead, our new approach trains a graph neural network (GNN) to determine mesh point locations by directly minimising the FE solution error from the PDE system Firedrake to achieve higher solution accuracy. Our GNN architecture closely aligns the mesh solution space to that of classical meshing methodologies, thus replacing classical estimates for optimality with a learnable strategy. This allows for rapid and robust training and results in an extremely efficient and effective GNN approach to online r-adaptivity. Our method outperforms both classical, and prior ML, approaches to r-adaptive meshing. In particular, it achieves lower FE solution error, whilst retaining the significant speed-up over classical methods observed in prior ML work.

Section: Introduction
Finite element methods (FEM) are currently the most widely-used tool for the large scale solution of partial differential equations (PDEs) (Ainsworth & Oden, 1997;Cotter, 2023). Central advantages are robustness, reliable error estimates, and thoroughly developed code bases (such as deal.II (Africa et al., 2024), DUNE (Bastian et al., 2008), Fenics (Logg et al., 2012), and Firedrake (Ham et al., 2023)), which are highly parallelisable and efficient. However, even with such optimised software the simulation of large scale problems (e.g. weather forecasting, structural simulations in engineering systems) is computationally costly. An important ingredient that determines the cost is the number of degrees of freedom (DOFs) required by a FEM to satisfy a chosen error tolerance. Since this cost depends on the number N z of mesh points z (i) of the underlying computational mesh, it is desirable to keep N z moderate. Mesh adaptivity based on mesh refinement and/or relocation to capture important solution features at the right scale can balance computational cost and accuracy. However, classical mesh-adaptive methods can be difficult to implement and require significant computational resources. In contrast, in the present work, we introduce a cheap, stable and highly efficient graph neural network (GNN) architecture to implement learnable mesh relocation (r-adaptivity). Our method keeps N z fixed and adapts the mesh point locations to reduce the overall FE error. Many classical mesh relocation methods have focused on finding and minimising mathematical substitutes of the FE error (usually simplified upper bounds) and solving additional (differential) equations to relocate the mesh points. For example, one can solve the Monge-Ampère (MA) equation (Budd et al., 2009) to find mesh point locations that minimise the interpolation error, which is an upper bound (up to some parameters and constants) of the FEM error arising from Céa's lemma (Huang & Russell, 2011). Recent Machine Learning (ML) approaches rely on similar mathematical simplifications and learn a surrogate for the mesh equations (Song et al., 2022;Zhang et al., 2024) leading to significant speed-up with comparable error reduction. In the present work we take an entirely different approach. We present G-adaptivity, an approach to mesh adaptivity that trains a GNN to generate meshes that directly minimise the error of the corresponding FEM solution. We couple backpropagation through a novel diffusion-based GNN-deformer, with mesh point gradients obtained through an application of Firedrake adjoint (Mitusch et al., 2019;Ham et al., 2023), to minimise the FEM approximation error directly (as opposed to the upper bound considered in (Huang & Russell, 2011)). The result is a model capable of outperforming the current state-of-the-art r-adaptive methods, whilst retaining the significant acceleration of ML based approaches (cf. Figure 1).
Figure 1. Optimised meshes from our new approach (G-Adapt) on the example of Poissons' equation: the error reduction (ER) achieved by classical Monge-Ampère (MA) can be significantly improved with direct optimisation (DirectOpt) of the FEM loss with respect to the mesh points, but at prohibitive additional cost. Our new approach achieves near optimal meshes in a fraction of the inference time.
Contributions Our work improves earlier ML based approaches to mesh relocation in the following ways:
• A novel training mechanism capitalising recent advances in FEM systems, which leads to a fast meshing algorithm that reduces the FE error even over stateof-the-art classical meshing methods. This was not possible in any prior surrogate ML approach;
• An improved GNN architecture based on a diffusion deformer, which allows for improved mesh relocation quality and provable non-tangling of generated meshes;
• A novel equidistribution loss regularizer, which enforces mesh regularity in unsupervised GNN training;
• Thorough numerical comparison with classical and recent approaches in terms of accuracy, mesh quality and computational time. Our experiments include both stationary and time-dependent test cases.
this section cite: ['b1', 'b18', 'b0', 'b5', 'b42', 'b7', 'b12', 'b33', 'b52', 'b58', 'b45', 'b7', 'b33']

Section: Related work
The effective approximation of PDE solutions is one of the central problems in computational mathematics. Over the recent decade, extensive work has been devoted to using ML for the numerical approximation of PDEs. This includes physics informed neural networks (PINNS) (Raissi et al., 2019;Raissi, 2018), Fourier Neural Operators (FNOs) (Li et al., 2020b;2023), graph neural operators (Li et al., 2020a), DeepONets (Lu et al., 2021), Message Passing Neural PDE Solvers (Brandstetter et al., 2022) and the deep Ritz method (E & Yu, 2018). The majority of such approaches try to directly approximate the PDE, or the associated solution operator, with a machine learning surrogate. Such methods offer certain advantages (for example in high dimensional settings (Han et al., 2018)), but are typically outperformed by traditional numerical methods in accuracy in most settings (Grossmann et al., 2023). Our approach is different. We use ML as a central ingredient of a finite element discretisation to construct an improved computational mesh, which is then coupled to a classical PDE solver. The crucial advantage is that we retain convergence guarantees and robustness of FEMs, something that is often lacking in direct ML-based PDE approximations. At the same time our approach achieves a significant speed up in the calculation of the improved mesh compared to classical approaches.
Adaptive mesh methods are a widely used tool for improving the performance of a classical FEM by varying the local density of the mesh points. This is necessary if the PDE solution has small length scales or singularities. Adaptivity allows achieving high accuracy without resorting to uniform mesh refinement. The most popular form is h-adaptivity (Ainsworth & Oden, 1997), in which mesh cells are subdivided when an a-posteriori estimate of the solution error is large. Such methods have complex data structures (see e.g. (Burstedde et al., 2011)) and, possibly, poor mesh regularity. Alternatively, the relocation based r-adaptive methods considered in this paper move a fixed number of mesh points to achieve a high density of points where a monitor m(z) of the solution error is large. Done correctly this can lead to significant error reduction but at some extra cost (Huang & Russell, 2011).
GNNs are the dominant approach to applying machine learning to irregularly structured data (Bronstein et al., 2017;Battaglia et al., 2018). There has been a proliferation of architectures inspired by spectral graph theory (Defferrard et al., 2016), convolutional (GCN) (Kipf & Welling, 2022), message passing (MPNN) (Gilmer et al., 2017) and atten-tional (GAT) (Veličković et al., 2018) approaches. More recently a range of differential equation inspired architectures (Chamberlain et al., 2021b;a;Giovanni et al., 2023) apply analytical tools to solve known problems with GNNs including stability, over smoothing and bottleneck phenomena. This algorithmic alignment along with the powerful message passing paradigm provide new solutions to some of the most pressing problems in science, including protein folding (Jumper et al., 2021), weather prediction (Lam et al., 2023), dynamics learning (Pfaff et al., 2023) and new numerical PDE solvers (Brandstetter et al., 2022;Lienen & Günnemann, 2022;Alet et al., 2019).
Fast ML based methods reduce the significant computational cost of classical methods for adaptive meshing. This includes work on h-adaptive mesh refinement (Foucart et al., 2023;Freymuth et al., 2023), and many contributions to radaptivity based on surrogate ML solvers of classical mesh movement PDEs (Yang et al., 2023;Hu et al., 2024) and supervised learning for mesh adaptivity using Graph Neural Networks (GNNs) (Song et al., 2022). A notable recent development is the universal mesh movement network (UM2N) (Zhang et al., 2024), which achieves error reduction on par with MA, but at significant speed up, and can also be applied to multiply connected domains.
this section cite: ['b50', 'b49', 'b18', 'b43', 'b9', 'b20', 'b31', 'b26', 'b1', 'b13', 'b33', 'b10', 'b6', 'b19', 'b35', 'b24', 'b55', 'b15', 'b25', 'b34', 'b36', 'b46', 'b9', 'b40', 'b2', 'b22', 'b23', 'b57', 'b32', 'b52', 'b58']

Section: Preliminaries and background

this section cite: []

Section: Problem specification
We consider finite element solutions to nonlinear secondorder PDEs on general domains Ω of dimension d. In abstract form, we can write these PDEs as follows:
F(u t , u, ∇u, ∇ 2 u) = f in Ω, αu+β∂ n u = g on ∂Ω,(1)
where α, β ∈ R. For transient problems, Ω = Ω × (0, T ), where Ω is a (d -1)-dimensional spatial domain and (0, T ) is the time-interval of interest. In this case, we employ the method of lines and combine the FEM with suitable timestepping schemes (Hairer & Wanner, 1996). To compute finite element solutions, we introduce a mesh T of the spatial domain with N z nodes, which we collect in the node set Z. The mesh T is used to construct trial and test functions with local support to discretize (1). For example, for a Poisson problem with homogeneous Dirichlet boundary conditions we consider the space of piecewise linear functions (vanishing on ∂Ω) S Z on T and solve: Find U Z ∈ S Z such that
(∇U Z , ∇v) L 2 (Ω) = (f, v) L 2 (Ω) ∀v ∈ S Z ;
(2) where (•, •) L 2 (Ω) denotes the inner-product in L 2 (Ω).
To minimise the error E(Z, U Z ) between the exact solution u of (1) and its finite element approximation U Z , r-adaptive meshing modifies the location of the node coordinates. Often, and in this work, r-adaptivity is particularly concerned with the reduction of the squared L 2 -error
E(Z, U Z ) := ∥U Z -u∥ 2 L 2 (Ω) .(3)
For transient problems, we tacitly assume that (3) is evaluated at the final time t = T .
this section cite: ['b27']

Section: Adaptive Meshing
Relocation based r-adaptivity turns a mesh with a certain topology into another mesh with the same topology. For this, the mesh points z (i) are moved, but their connectivity (and hence the associated data structures) is unaltered. Such methods typically map a fixed mesh in a computational domain (i.e. a representation of the mesh graph) to a deformed mesh in the physical domain where the PDE is posed. Note that, for transient problems, Ω should be replaced by Ω and d should be replaced by d -1 in the following explanation (cf. Section 3.1). We denote the mesh points in the physical domain by Z = {z (i) } Nz i=1 which form a triangulation T of Ω with N T mesh elements, i.e.
T ⊂ ∆ (j) ⊂ Z; |∆ (j) | = d + 1 , |T | = NT .
We define the following domains and coordinates: the "computational" domain Ω C is mapped to the "physical" domain Ω P ⊆ R d , and the "computational" coordinates ξ ∈ Ω C are mapped to the "physical" coordinates z ∈ Ω P . To construct an adaptive mesh we consider a differentiable, possibly time-dependent, deformation map F : Ω C → Ω P , so that z = F(ξ, t) and F(∂Ω C ) = ∂Ω P . If ξ (i) are the fixed mesh points in the computational domain then z (i) = F(ξ (i) , t). Assuming the mesh in the computational domain is regular, then determining the (properties of the) mesh in the physical domain, reduces to finding, (and analysing), F.
Location based methods find F by solving a PDE, or a linked variational principle. Monge-Ampére (MA) methods assume that F is a Legendre transform with a 'mesh potential' ϕ(ξ, t) for which F = ∇ ξ ϕ. The linearisation of F is given by J = ∂F/∂ξ ≡ H(ϕ) where H is the Hessian of ϕ. Relocation methods usually equidistribute a monitor function m(z) so that ϕ satisfies the MA equation
m(z)|H(ϕ)| = m(∇ϕ)|H(ϕ)| = θ, for constant θ.
(4) For example, in (Huang & Russell, 2011) m(z) is an apriori monitor of the interpolation error. The PDE (4) has a unique, convex, solution (Budd et al., 2013) which avoids mesh tangling. However, (4) is expensive to solve and, in its pure form, only applicable to simply connected domains. Solution procedures include relaxation methods (Budd et al., 2009), quasi-Newton methods (McRae et al., 2018), surrogates (Song et al., 2022;Zhang et al., 2024), and PINNs (Yang et al., 2023).
Velocity based methods find an ODE describing the mesh point evolution in pseudo-time τ so that ∂z (i) /∂τ = v(z (i) , t).
(
The choice of velocity function v is critical to the success of such methods, and is often motivated by natural Lagrangian structures of the underlying PDE. These methods provide the basis of our diffusion-based deformer (diffformer) described in section 4.1 and while they often lead to mesh tangling where mesh lines cross (cf. Ch. 7 in (Huang & Russell, 2011)), our architecture is specifically designed to enforce non-tangling of the mesh (cf. Theorem 4.2).
this section cite: ['b33', 'b11', 'b12', 'b44', 'b52', 'b58', 'b57', 'b33']

Section: The G-adaptivity framework
The G-adaptive mesh relocation method described below is essentially a velocity based method with learnable coefficients that are trained by calculating the rate of change of the FE solution error E with respect to the mesh point location. As we explain below, the G-adaptivity framework combines feature selection, structural regularization and direct optimisation to learn optimal mesh relocation in an unsupervised manner whilst avoiding mesh tangling.
this section cite: []

Section: Graph-based adaptive mesh refinement
For simplicity of exposition we focus our discussion on the 2D case, but note that this approach generalises in a straightforward manner to 3D cases as shown in Section 5.5. A mesh T (i.e. a triangulation of the domain Ω) with meshpoints Z gives rise to a natural graph, with the nodeset V = Z and the edgeset E = {(z i , z j ) ∈ V × V; ∃∆ ∈ T , s.t. z i , z j ∈ ∆}, i.e. two nodes share an edge if there is a triangle in the mesh T which has both nodes as vertices. The graph (V, E) can be enriched with node features x i ∈ R d0 : i ∈ V represented in matrix notation as X ∈ R Nz×d0 . For example we could associate to each mesh point z (i) the value of the solution field u(z (i) ) as a feature. Likewise, we can introduce latent features that propagate through repeated application of a map on the graph, this is used in our architecture (cf. Figure 2). Mesh connectivity is stored in the adjacency matrix A (where
a ij = 1 if (i, j) ∈ E and zero otherwise). A graph neu- ral network (GNN) M θ : R Nz×d0 × E → R Nz×d N
is a map from features to features constructed with layers L θ k : R Nz×d k → R Nz×d k+1 acting node wise as
x k+1 i = L θ k (x k i ) = ϕ θ k   x k i , j∈Ni φ θ k (x k j )  
where φ θ k is the learnable edge-wise operation, ϕ θ k is a learnable node-wise aggregation and N i = {j ∈ V; (i, j) ∈ E} is the set of nodes adjacent to the meshpoint i.
Integral to the G-adaptivity framework is the construction of the feature matrix such that the GNN can act as a mesh deformer. Similar to (Zhang et al., 2024) we construct the feature matrix by concatenating coordinates of a regular mesh ξ ∈ R Nz×d with a learnable feature encoding h θ (Z 0 , H) which, motivated by ( 4), is dependent on the Frobenius norm of the Hessian H(U Z 0 ) = ∥∂ i ∂ j u∥ F : Ω → R of the FEM solution U Z 0 on the undeformed mesh Z 0 . When higher order finite element functions are used in the approximation space of U Z the Hessian can be obtained by simple differentiation, but even in the case of linear elements this information is recoverable using widely-used techniques such as the one described in Appendix A.2. The final input feature matrix is then
X = Z 0 ∥ X 0 λ ∈ R Nx×(d+|λ|) , X 0 λ = h θ (Z 0 , H),
where λ denotes the index set for the node features, which is passed into a GNN mesh deformer that then outputs the relocated mesh points. Previous works (Song et al., 2022;Zhang et al., 2024) used a graph attention network (GAT) (Veličković et al., 2018)
as the GNN mesh deformer Z k+1 X k+1 λ = A θ (X k )Z k σ λ A θ (X k )X k λ W λ(6)
where X = (Z, X λ ) and W λ is a learnable linear transformation matrix. To prevent mesh crossing the non-linearity and channel mixing are excluded from the positional channels in (6). In the above A θ (X k ) is row-stochastic meaning that the top row of ( 6) corresponds to a graph-based averaging over graph neighbours. Motivated by (Chamberlain et al., 2021b) and velocity-based methods for meshpoint relocation introduced in section 3.2, in our G-Adaptive framework this average is replaced by a diffusion based deformer (henceforth referred to as Diffformer)
Ż(τ ) = (A θ (X k ) -I)Z(τ ), Z(0) = Z k ,(7)
which is solved to a finite end time τ = τ end and leads to the meshpoint update Z k+1 = Z(τ end ), i.e. an overall deformer of the form
Z k+1 X k+1 λ = Z(τ end ) σ λ A θ (X k )X k λ W λ .(8)
As before, the learnable attention A θ is row-stochastic, meaning ( 7) is essentially a diffusion equation on the graph V. Further details on our Diffformer are provided in Appendix A.1. We can stack multiple layers of (8), each time varying the number of hidden feature dimensions which are updated using the second row of (8). We denote the overall GNN by the map M θ and a schematic overview of the components of M θ is provided in Figure 2.
this section cite: ['b58', 'b52', 'b58', 'b55', 'b15']

Section: Structural regularization
The architectural changes between ( 6) and ( 8) lead to several regularity properties that we refer to as structural regularization. The Diffformer based architecture has a key advantage
Input mesh coordinates Z in = (z (i) in ) Nz i=1 Curvature information: H(U Zin ) Feature extractor X 0 λ = h θ (Z in ; H) Diffformer: Ż = (A θ (X k ) -I)Z Computation of attention → A θ (X k ) Feature deformer: X k+1 λ = σ λ A θ (X k )X k λ W λ Deformed mesh coordinates Z out = (z (i) out ) Nz i=1 + + Next layer: k → k + 1 Figure 2. Schematic overview of our new graph diffusion-based architecture.
over other velocity based methods in the generation of regular meshes. A requirement of FEM meshes is that they are not 'tangled', i.e. that they form a well-posed triangulation of the domain Ω (i.e. no triangles overlap). This follows if each mesh point is in the interior of the convex hull of its neighbours on the graph and can equivalently be characterised using the Jacobian of the deformation map M.
Definition 4.1. Let J (i) be the Jacobian of the deformation map M at simplex ∆ (i) . A mesh is said to be tangled if there exists a simplex where the determinant of the Jacobian det(J (i) ) ≤ 0 (Huang & Russell, 2011).
Velocity based methods often lead to tangled meshes due to the local way in which the mesh point movement is defined. However, this does not arise in our method.
Theorem 4.2 (Discrete-Time Non-Tangling). If the diffusion equation ( 7) is solved with the forward Euler method, then for sufficiently small pseudo-timestep dτ < 1/2, the discrete mesh evolution under the deformation map M preserves element orientations, ensuring that no mesh tangling occurs.
A full proof is given in Appendix F but in essence the diffusion process ensures that meshpoints are simultaneously moved along directions that point into the convex hull of neighbouring meshpoints, thus ensuring that tangling cannot occur (cf. Figure 3).
The proof relies on the softmax of the attention mechanism normalising the adjacency to be row stochastic and for the time step of the residual connection to be controllable. This is a benefit over (6) and allows (8) to learn an anisotropic diffusion which is akin to a learnable monitor function from
xi(t) (AX (t))i Diffformer xi(t + δt)
Figure 3. The action of the graph diffusion pulls nodes into the convex hull of their graph neighbours.
classical relocation methods.
this section cite: ['b33']

Section: Firedrake adjoint optimal gradient computation and direct FEM loss
Training the GNN M θ requires computing the derivative of E(Z) = E(Z, U Z ) with respect to the node coordinates Z. Since evaluating E(Z, U Z ) requires solving the PDE (1) first, a naïve application of automated differentiation would result in the solution of additional (N z × d) PDEs (N z × (d -1) in the transient case). To avoid the additional computational cost, we employ the well-established method of adjoints. Specifically, we employ Firedrake's automated adjoint capabilities implemented in pyadjoint (Mitusch et al., 2019). With pyadjoint, derivatives with respect to mesh coordinates can be computed in an automated fashion as shape derivatives of E(Z, U Z ) in directions discretized with vector-valued linear Lagrangian FEs (Ham et al., 2019a). Remark 4.3. Often, the exact solution u to ( 1) is not known and must itself be approximated with the FEM (e.g., by interpolating onto U Z a FE solution computed on a much finer mesh). In this case, it is essential to correct the directional derivatives computed with Firedrake by adding the corrections terms stemming from evaluating the formula
Ω (u -U Z )∇u • V dx along each finite element direction V ∈ S d Z (V ∈ S (d-1) Z
in the transient case). This is necessary because the shape derivative of a FE function in a direction discretized with FEs is zero (Ham et al., 2019a).
this section cite: ['b45']

Section: Regularized gradients
In line with the concept of equidistribution discussed in Appendix E.2 we introduce a regularizing term in training which further enforces mesh regularity in an unsupervised manner and leads to improved training of the mesh deformer using only information about a predefined monitor function m(U Z ) (we follow (Zhang et al., 2024) and use m(U
Z ) = 1 + 5 |∇U Z | maxΩ |∇U Z | ).
The motivation is to provide a global signal that moves mesh points into regions of the domain where the solution varies and likely requires more meshpoints to resolve. For this we add the following regularizing term to our loss:
L equi (Z) = ∆ (j) ∈T ∆ (j) m(x)dx -m 2 , where m = |T | -1 ∆ (j) ∈T ∆ (j) m(x)dx.
Given the area of a simplex in the mesh α(∆ (i) ) ∈ T the terms in the above loss are approximated by ∆ (j) m(x)dx ≈ α(∆ (j) )c d z∈∆ (j) m(z), where c 2 = 1/3, c 3 = 1/4. This leads to the following full regularized loss which we use in the training of our Diffformer:
L θ = E(M θ ) + L equi (M θ )(9)
During training the weighted graph Laplacian (A θ -I in ( 7)) will adaptively adjust to minimize both terms, meaning the mesh evolution will not purely follow the degree weighted graph Laplacian dynamics but will now be biased towards error reduction and equidistribution.
this section cite: ['b58']

Section: Experimental results
We evaluate G-adaptivity on three classical meshing problems in two-dimensions: an elliptic PDE (Poisson's equation) in a variety of convex domains, a nonlinear timeevolution PDE (Burger's equation), and the time dependent Navier-Stokes equations in a multiply connected domain.
In the following we present the performance improvements obtained in terms of the FEM L 2 -error reduction (cf. (3)) and compute time, using our novel approach for adaptive meshing on each of these problems. Additional experiments and sensitivity analysis is provide in the Appendix D. Full code to build the datasets and reproduce our results can be found at https://github.com/JRowbottomGit/g-adaptivity.
this section cite: []

Section: Experimental details
Method Our experimental pipeline consists of three parts: (i) we build datasets containing information about the PDE, FEM solution on a regular (i.e. not relocated) grid and the corresponding approximation of the Hessian of the solution; (ii) then we train either our model or the baseline to predict a relocated mesh on which we perform another FE solve to obtain the improved solution approximation; (iii) finally, we compare this FEM solution to a reference solution (calculated on a fine reference mesh) and determine the change in L 2 -error over the original undeformed mesh, i.e. in the above notation we look at relative error reduction of E(U M θ ) over E(U Z0 ). These steps are repeated for the three PDE datasets as described below. Note that the Firedrake adjoint solve is only required during training of our G-Adaptive network and not during inference. The times reported in the below numerical examples thus contain a true reflection of the fast online mesh adaption times.
Datasets For each experiment described below we build a randomised dataset with held-out test data. This means in each case we specify a set of solution values through varying source terms or boundary conditions (adjusted to the PDE at hand). We then generate training and test sets of coarse, undeformed, meshes T of varying resolution with associated FEM solution values and, for each mesh, we also compute a reference solution (on a finer reference mesh) which serves as comparison for the error computation. For most test cases the solution values are generated by randomly sampling Gaussians in the domain Ω. For the example of the Navier-Stokes flow around a cylinder we used snapshots of a timeseries simulation of vortex shedding. Full details on the specific configurations for each experiment are provided in Appendix C.
Baselines We compare our algorithm against two adaptive mesh algorithms: classical MA as described and implemented by (Wallwork et al., 2024) and the ML based surrogate GNN method UM2N (Zhang et al., 2024). These two state-of-the-art approaches serve as a baseline for the FEM error reduction and deformation time. As a third baseline we train UM2N on our regularized PDE loss (9), which we denote by UM2N-G in the tables below, in order to highlight the performance improvements gained from both our new architecture (Figure 2) and our new training ( 9).
this section cite: ['b56', 'b58']

Section: Experiment details
Here we refer to our framework Gadaptivity and our model Diffformer synonymously, which we train using the regularized PDE-loss (9) (which is the regularized L 2 FEM approximation error + equidistribution regularizer). Calculation of the L 2 -error is obtained by calculating an FE solution on the moved mesh and comparing this to the projection of the reference solution of a fine regular reference mesh onto sufficiently higher order elements.
We train our new model (G-Adapt) and UM2N-G for 300 epochs using an Adam optimiser and learning rate of 0.001. Our model has 4 diffformer blocks as described in Appendix C. Each blocked is rolled out using explicit Euler integration for 32 steps with a step size of 0.1. For the baseline UM2N we trained using 1000 epochs in order to achieve good performance but we believe further tuning of the training may be required in order to achieve a similar performance to the one reported in (Zhang et al., 2024). UM2N remains an important baseline and we expect that with appropriate training the method would be able to achieve a similar error reduction (ER) as the Monge-Ampère (MA) solver, but we would like to highlight that even in the best reported results of the original paper, UM2N never achieved a larger error reduction than MA.
Evaluation We report three metrics to evaluate the performance of the mesh relocation methods at hand: (i) the relative L 2 error reduction (ER) of the FE solution on the relocated mesh versus the FE solution on the initial coarse mesh (larger error reduction means improved performance);
(ii) the time taken to relocate the mesh (shorter times means faster relocation); (iii) the aspect ratio of the deformed mesh as a measure of mesh quality as described in Appendix F.4 (a single digit aspect ratio is generally acceptable and a smaller aspect ratio indicates a more regular mesh). Each experiment is performed in full five times (training and evaluation) with different random seeds to provide the error bars.
this section cite: ['b58']

Section: Benchmarking on Poisson's equation
Our first benchmark is on the classical Poisson problem -∇ 2 u = f (z) with Dirichlet boundary conditions. Full details of the FEM formulation are given in Appendix C.1. We benchmark the results against two datasets on a square (cf. Figure 1) and polygonal domain respectively (cf. Figure 4). Further evaluations on five additional (nonconvex) geometries are provided in Appendix D.4. For both aforementioned examples we sample source terms and boundary conditions corresponding to underlying Gaussian fields. On the square domain Ω = [0, 1] 2 we initialise the mesh-deformation with a regular grid and to showcase G-adaptivty's ability to work on irregular domains with unstructured meshes we apply a similar methodology to the convex polygonal dataset (a sample is shown in Figure 4). The results for both datasets are presented in Table 9. The central observation in these results is that our methodology provides the very first ML approach to mesh relocation which is able to outperform MA in terms of error reduction, while retaining the fast mesh relocation times given by the state-of-the-art GNN -UM2N (Zhang et al., 2024). † The direct optimization method is included here purely for exposition, showing that MA-meshes are not necessarily optimal. DirectOpt computes the optimal mesh for a given PDE with known solution but is extremely slow and relies on data which is not available during inference, thus it does not constitute a practical adaptive meshing strategy. In contrast, once trained, our G-Adaptive approach yields fast online mesh movement without needing reference solution values.
this section cite: ['b58']

Section: Time-dependent Burgers' equation
In our second example we highlight that our approach can equally well be applied to time-dependent problems, in particular the viscous Burgers' equation:
∂u ∂t + (u • ∇)u -ν∇ 2 u = 0.
Further details on the specific FEM implementation (and implicit time-stepper) used are given in Appendix C.2. We randomly sample Gaussians on the square domain Ω = [0, 1] 2 as initial conditions for the evolution in Burgers' equation and perform the following experiments.
Burgers' square rollout: We train the models on a set of Gaussian initial conditions for a timestep δt = 0.02 with 2 steps and evaluate by following 10 trajectories of randomly sampled Gaussians in the Burgers equation for 20 timesteps, remeshing after every 2 steps (cf. Figure 5 and Appendix D.3). The results in the top part of Table 2 show the average error reduction over achieved over every block of two timesteps. While the MA performs well on this task, we note that the UM2N and UM2N-G baselines appear to lead to a negative error reduction (i.e. an increase), which is likely due to the fact that the Burgers' equation changes the solution shape and thus trajectories will lead to out-of-distribution cases for methods that are trained only on initial conditions. Due to the structural regularity of our new approach (cf. Section 4.2) our approach is able to deal with out-of-distribution data very well, and most importantly is able to outperform MA in terms of error reduction while retaining a fast mesh relocation time.
Burgers' square 10 steps: The interpolation error in remeshing is significant and provides a central limitation to current mesh relocation techniques (cf. (Budd et al., Figure 5. Snapshots of the velocity field (x-component) together with the corresponding deformed meshes provided by Monge-Ampère (MA) with 46.52% average error reduction over the full solution path compared to the deformed meshes provided by our approach (G-Adapt.) with 49.15% error reduction. 2009)). It is thus desirable to relocate meshes only after several timesteps. It turns out that our approach lends itself to targeted training not just of a GNN that reduces the FEM error in a stationary sense, but a GNN that seeks to find an optimal mesh given a specified remeshing frequency. The classical method MA has no means of inferring this information or adjusting the meshes accordingly. On this example we trained the GNN on a collection of random Gaussian initial conditions with the loss attained by solving the corresponding FEM problem for 10 timesteps of size δt = 0.02. The results in Table 2 highlight that in this way we can achieve even more significant ER over MA thus leading to efficient meshes that require less frequent changes in time-evolving systems.
this section cite: []

Section: Navier-Stokes equation and flow past a cylinder
Our final example is the canonical flow past cylinder problem we simulate data using an FE solution for 400 time steps of size δt = 0.01 of the time series evolution expressed in Gaussian basis function expansions (cf. Figure 6 and Appendix A.3). The training and test data are 25 and 50 respectively random snapshots from the range t ∈ [1, 4] with remeshing after every 5 timesteps. Full details of the PDE and FEM formulation are provided in Appendix C.3. Again we observe good error reduction and fast mesh relocation times in our new methodology.
this section cite: []

Section: 3D adaptive meshing
The G-Adaptivity framework and diffusion deformer model are also easily adapted to the 3D setting. To demonstrate this we perform an experiment on a 10x10x10 unit cube for the 3D Poisson problem with Dirichlet boundary conditions and Gaussian solutions, analogous to Section 5.2. An example of the corresponding results can be seen in Figure 7. In the interest of brevity, the full numerical results are presented in Appendix 5.6 and show that the method outperforms MA significantly (out-of-the-box UM2N does not apply in 3D) and that it leads to effective mesh point concentration in regions of interest.
this section cite: []

Section: Scalability of the G-Adaptivity framework
The G-Adaptivity framework is able to scale to very large meshes. In particular the inductive learning property of GNNs ensures the ability of GNNs to transfer to unseen graphs in this case meaning we can perform super-resolution.
In Table 9 we report experiments where the model is trained on 15x15 mesh and inference is performed on larger 60x60 (3,600 nodes) and 150x150 (22,500 nodes) meshes for the Poisson problem with 128 sampled Gaussians (see Figure 8). In order to scale the transformer encoder, which in naive form scales with O(N 2 ) edges we use a sliding window SWIN (Liu et al., 2021) style transformer to capture the monitor function embedding at the mid-length scales.
Our model consistently achieved significant mesh adaptation, accuracy improvement, and computational acceleration compared to Monge-Ampére, matching the performance observed on smaller-scale experiments. We have presented a novel, and effective, approach to the classical problem of r-adaptive meshing for FEM solutions of PDEs. In particular, we demonstrate, that GNNs together with a differentiable FEM solver (Firedrake), and a loss function given by the regularized solution error, can be effectively used to optimise the location of mesh points to minimise the FEM error. Hence we can take an entirely different route from prior work (both classical and ML approaches) which determine good choices of mesh points by analysis-inspired heuristics using a location based approach. We demonstrate the advantages of our method on challenging test problems in two dimensions, including in a multiply connected domain, and find that, on those examples, we are able to outperform both classical and ML methods in terms of error reduction while retaining similar computational cost to prior ML work. We note that the direct FEM error optimisation approach extends naturally to more complex domains, and PDEs, where classical methods may struggle providing a basis for future extensions of this work.
Finally, we note that any machine learning-based approach is inherently statistical in nature, meaning that GNNbased meshing tools are likely to perform worse on out-ofdistribution data. We observed this in our experiments with both pre-trained UM2N models and our own G-Adaptive approach when applied to PDEs whose solutions exhibited markedly different scales and features from those seen during training. Enhancing the scale-generalisation capabilities of ML-based adaptive meshing therefore remains an important open problem for future investigation.
this section cite: ['b41']

Section: A. Implementation details A.1. Diffusion deformer (diffformer) details
We apply the diffformer in learned blocks
D (b) θ (X) = T b n=0 (I + dt(A (b) θ (X (b) ) -I)) X (b) .(10)
where n = 0, . . . , T i /dt denotes the discrete time step index, N b is the number of blocks in the deformer, such that A
θ (X) is the attentional adjacency matrix at block b, dynamically learned as:
a (b) ij = exp(ϕ (b) θ (X i , X j )) k∈N (i) exp(ϕ (b) θ (X i , X k ))
.
this section cite: []

Section: Then the full G-adaptivity diffusion based deformation Map is given by
M θ (X 0 , A) = N b b=0 D (b) θ X (n) ,(11)
The process consists of: 1. Initializing the feature matrix X (0) as the feature positions. 2. Looping over N b deformer blocks, updating positions iteratively. 3. Applying T i /dt steps of discrete evolution to refine the mesh over time.
The input feature matrix is X 0 = (ξ ∥ h) ∈ R Nx×d+|λ| utilises the graph transformer encoder of (Zhang et al., 2024) with the exact same hyperparameters. Similarly each attentional matrix A
θ is adapted from the same. We use N b = 4 blocks and rollout using explicit Euler time integration for 32 timesteps with a step size of 0.1.
this section cite: ['b58']

Section: A.2. Hessian recovery
To identify parts of the domain Ω where the solution varies rapidly in space, we use an estimator for the local Hessian H(x, y) which is inspired by the approach in (Picasso et al., 2011). For a piecewise linear function u ∈ V (Ω) an approximation of the components of H is obtained by solving the weak problem
- Ω ∂ i u∂ j v dx = Ω H ij v dx for all v ∈ V (Ω), v| ∂Ω(12)
for H ij subject to the strong Dirichlet boundary condition H ij | ∂Ω = 0. While there might be other Hessian recovery techniques (see e.g. (Vallet et al., 2007)), we observe empirically that our approach leads to good results if the Frobenius norm ||H|| F = i,j H 2 ij is fed as an input to the GNN.
this section cite: ['b47', 'b54']

Section: A.3. Gaussian basis function expansion for time-dynamic training
For technical reasons, in the Navier Stokes dataset it was necessary to provide the initial conditions used for training in analytical form as an UFL (Alnaes et al., 2014) expression that can be fed to Firedrake. To achieve this, snapshots of the pressure and velocity fields are taken at specified times during the numerical solution of the time-dependent Navier Stokes equations. The fields w(x, y) are approximated as a sum of Gaussian basis functions in the form
w GBF (x, y) = ij a ij ϕ(x -x i , y -y j ) with ϕ(x, y) = exp - 1 2 x 2 h 2 x + y 2 h 2 y (13
)
where the n x × n y = 8 × 8 nodal points (x i , y j ) are arranged in a regular Cartesian grid over the domain with grid spacings h x and h y . The expansion coefficients are chosen such that w GBF (x i , y j ) = w(x i , y j ). The sum on the right hand side of (13) can be implemented as an UFL expression.
conditions u| ∂Ω = 0 such that Ω ∇u • ∇v dx = Ω f v dx for all test functions v ∈ S Z , v| ∂Ω = 0.
this section cite: ['b3']

Section: C.2. Burgers' equation
The non-linear viscous Burgers' equation describes the evolution of the vector-valued velocity field u as ∂u ∂t
+ (u • ∇)u -ν∇ 2 u = 0 in Ω,(15)
where ν > 0 is the kinematic viscosity and we solve consider a two-dimensional rectangular domain Ω ⊂ R 2 . The term (u • ∇)u describes non-linear convection and ν∇ 2 u is the viscous diffusion.
We use a piecewise linear Finite Element discretisation with u ∈ S d-1 Z . A simple backward-Euler timestepping method with step-size ∆t is employed to compute the velocity u n+1 ∈ S d-1 Z at the next timestep from the current velocity u n ∈ S d-1
this section cite: []

Section: Z
The time-discretised weak form of (15) is given by:
find u n+1 ∈ S d-1 Z such that Ω u n+1 -u n ∆t • v + (u n+1 • ∇u n+1 ) • v + ν∇u n+1 : ∇v dx = 0 (16
)
for all test functions v ∈ S d-1 Z . The final two terms in ( 16) are the weak form of the nonlinear advection and viscous diffusion term respectively.
this section cite: []

Section: C.3. The Navier-Stokes Equations
We consider the incompressible Navier-Stokes equations in primitive form for a time-dependent velocity field u and pressure p in the two-dimensional spatial domain Ω = [0, 2.2] × [0, 0.41]:
∂u ∂t + (u • ∇)u -ν∇ 2 u + ∇p = f , in Ω,(17)
∇ • u = 0, in Ω,(18)
Here ν > 0 is again the kinematic viscosity and f is an external force term.
The Finite Element discretisation uses Taylor-Hood elements with piecewise linear pressure and vector-valued piecewise quadratic velocity functions (u, p) ∈ Q d-1 Z × S Z . The time-stepping procedure, which computes the velocity u n+1 ∈ Q d-1 Z and pressure p n+1 ∈ S Z at the next timestep from the current velocity u n ∈ Q d-1 Z , is a variant of Chorin's projection method (Chorin, 1967;1968). It consists of three steps, each of which requires the solution of a weak problem.
this section cite: ['b16', 'b17']

Section: Step 1: Compute tentative velocity u
* Find u * ∈ Q d-1 Z such that: Ω u * -u n ∆t • v + (u n • ∇u mid ) • v + ν∇u mid : ∇v dx + ∂Ω (p n n • v -ν(∇u mid • n) • v) ds = Ω f • v dx. (19
)
for all piecewise quadratic vector-valued test functions v ∈ Q d-1 Z where u mid = 1 2 (u n + u * ). Homogeneous Dirichlet boundary conditions are applied at the top (y = 0.41) and bottom (y = 0) of the domain. The velocity field is prescribed on the inflow boundary at the left side of the domain as u(x = 0, y) = 4.0 • 1.5 • y • 0.41-y 0.41 2 , 0 . The weak problem in ( 19) is solved with a GMRES iteration that is preconditioned with successive overrelaxation (SOR).
this section cite: []

Section: Step 2: Solve for pressure correction
To ensure that the velocity field at the next timestep is divergence-free, find p n+1 ∈ S Z which satisfies the following elliptic problem:
Ω ∇p n+1 • ∇q dx = Ω ∇p n • ∇q dx - 1 ∆t Ω (∇ • u * )q dx (20
)
for all piecewise linear pressure test functions q ∈ S Z . To deal with the fact that the pressure is only determined up to an additive constant, homogeneous Dirichlet boundary conditions are applied to p n+1 , q at the outflow boundary. The weak problem in (20) is solved with a conjugate gradient iteration preconditioned with algebraic multigrid (AMG).
this section cite: []

Section: D.3. Additional examples of Burger's evolution
In Figure 11 we include some additional mesh trajectories from the experiment performed in Section 5.3.  an optimal value in the Poisson square example was a loss weighting of 1, which was used throughout all experiments.
this section cite: []

Section: D.6. Scalability of the G-Adaptivity framework
In Section 5.6 we claimed the G-Adaptivity framework is able to scale to very large meshes via super-resolution. In Table 9 we report experiments where the model is trained on 15x15 mesh and inference is performed on larger 60x60 (3,600 nodes) and 150x150 (22,500 nodes) meshes for the Poisson problem with 128 sampled Gaussians (see Figure 8). We also show the results for G-Adaptivity applied to a 10x10x10 (1,000 node) cube (see Figure 7). In particular it works with a constant data structure, is easy to use on parallel architectures, it gives a more regular mesh (often with guaranteed mesh regularity), it naturally inherits Lagrangian and scaling structures in a PDE (which is very useful for example in ocean modelling and studying PDEs with singularities), and can be easily linked to existing external software designed to solve a PDE on an unstructured mesh (for example a discontinuous Galerkin solver). As a result, r-adaptive methods have recently been very successfully used, for example, in the operational data assimilation codes of national weather forecasting offices, which when coupled to the computational dynamical core, have led to a very significant increases in computational accuracy, particularly for resolving local weather features such as fog and ice (Piccolo & Cullen, 2012). r-adaptivity has also found natural applications in the steel industry where the Lagrangian nature of the approach is very well suited to the evolving fine structures in the forging process (Uribe et al., 2024). Possible disadvantages of r-adaptivity, such as excessive mesh computation cost, and a tendency to mesh tangling, are exactly the issues we address in this paper, proposing a fast and accurate method which avoids tangling.
this section cite: ['b48', 'b53']

Section: E.2. The Equidistribution Principle
The equidistribution principle applied to a mesh with cells C i used for an FE calculation of a function u(z), aims to minimise the total error over all the cells by equidistributing it over each cell. Typically the error over such a cell can be measured (or estimated) by the integral of an appropriate monitor function over that cell, or more simply by the expression
m(z)|C i | (22
)
where z is a representative point in the cell, and (in the two dimensional case) |C i | is the cell area. The equidistribution condition on the cells C i then becomes m(z
)|C i | = θ,(23)
where θ (to be determined) is a constant. The function m is usually a function of u. An important example is given by the problem of linearly interpolating u(z) as it follows from Céa's lemma that the resulting interpolation error is an (often tight) upper bound for the FE solution error. In this case m will be a function of the curvature of u (with the exact form dependent on the norm used to measure the error) (Huang & Russell, 2011).
In the context of r-adaptivity each such cell C i in the physical domain, will be the image, under the action of the deformation map F of a reference cell (of fixed area) in the computational domain. The area |C| i of C i will then be proportional to det(J) where J is the Jacobian of F. The equidistribution condition (23) then becomes:
m(z) det(J) = θ.(24)
Note that the application of the monitor function in this way is equivalent to defining a measure on the physical space.
In one dimension the equidistribution condition (24) uniquely defines each cell length, and thus the cell shape, and hence the whole mesh. However in two dimensions it only gives the cell area but not the shape. To find the mesh uniquely additional conditions must be imposed. Noting the correspondence between the equidistribution condition and a measure on the physical space, the deformation map can be viewed as mapping a uniform measure in the computational space to a new measure in the physical space. It is natural to seek a map which minimises the cost of doing this, as this leads to meshes in the physical domain which are close to uniform and hence have minimal skewness and which avoid tangling. This gives an obvious link between mesh generation and optimal transport. In the continuous setting such a map can be calculated by solving (either directly or by using a surrogate solver) an associated Monge-Ampére equation, leading to the MA methods described in the main body of the text. Note that with modifications this procedure can also be used to generate meshes on non-planar manifolds (McRae et al., 2018).
this section cite: ['b33', 'b44']

Section: F. Mesh Tangling Prevention
We provide a formal proof that a mesh evolution scheme based on the row-stochastic weighted graph Laplacian does not lead to tangling, provided a sufficiently small time step is chosen. The argument follows from the positivity of the determinant of the Jacobian of the deformation, which is preserved due to the eigen-structure of the graph Laplacian.
Remark F.1 (Iterative Application in GNN Blocks). The below results extend to our GNN-based mesh deformer, which applies diffusion blocks iteratively. At each iteration, the network updates the node positions while resetting the adjacency weights and initial state X 0 . Since each block follows the same form the results can be applied recursively. This ensures that stability and mesh preservation hold across multiple diffusion steps, allowing controlled adaptation of the mesh throughout the G-adaptivity pipeline.
Definition F.2 (Weighted Random Walk Normalized Graph Laplacian). Given a weighted graph G = (V, E, A θ ) with adjacency matrix A and a learnable weight matrix A θ , where (A θ ) ij represents the weighted edge between nodes i and j, the weighted degree matrix is defined as D ii = j (A θ ) ij . The weighted random walk normalized graph Laplacian is given by:
∆ θ = I -D -1 A θ .
The operator ∆ θ is symmetric positive semi-definite, satisfying ∆ θ ⪰ 0. Its eigenvalues satisfy 0 = λ ∆ θ 0 ≤ . . . ≤ λ ∆ θ n-2 ≤ ρ ∆ θ , with ρ ∆ θ ≤ 2. The eigenvalues represent the graph frequencies, and the corresponding eigenvectors are denoted by {ϕ ∆ θ ℓ } n-1 ℓ=0 . The weights (A θ ) ij satisfy a i,j > 0 if (i, j) ∈ E and j∈N (i) a ij = 1.
• In the degree-normalised graph (random walk) Laplacian (A θ = A), row sums are preserved due to the degree normalization, ensuring j Ãij = 1, where Ã = D -1 A.
• In the softmax-weighted case, weights are computed as
(A θ ) ij = exp(f (X i , X j )) k∈Ni exp(f (X i , X k ))
, enforcing row stochasticity j (A θ ) ij = 1.
this section cite: []

Section: F.3. Monitor-Conditioned Time Step
To refine the time step bound, consider the propagation matrix M = I -dt∆ with eigenvalues µ i = 1dtλ i . The condition number of M is κ(M ) = 1-dtλmin 1-dtλmax . Similarly, by Gershgorin's theorem, λ max ≤ 2. Theorem F.6 (Monitor-Conditioned Time Step). Given the discrete update X k+1 = (I -dt∆)X k , where the monitor function redistributes the mesh to improve spectral conditioning, the time step satisfies dt ≤ min 1 2 , κ(M ) 2 .
Proof. The local mesh determinant propagates as J k+1 = det(I -dt∆)J k . Stability requires 1dtλ max > 0. Since λ max ≤ 2, we obtain dt ≤ κ(M ) 2 , completing the proof.
this section cite: []

Section: F.4. Mesh Quality Measures
Mesh quality measures are often used as indicators of whether a mesh will be effective when used to solve a PDE. In particular, the mesh-quality metrics are directly related to the conditioning of FEM stiffness matrices, meaning poor mesh conditioning leads to numerical instabilities in the FEM solvers. Our method is designed to minimise the FE solution error directly, but the inclusion of the equidistribution regularisation ensures that G-Adaptivity leads to meshes that maintain good conditioning while reducing the FEM error.In our numerical experiments we report aspect ratio as a strong indicator of this mesh conditioning, but in the relevant literature the following metrics are commonly used to often used to assess mesh scale, skewness, and regularity.
Two paradigms exist for evaluating mesh quality:
• Known deformation map: Mesh quality is assessed directly using the eigenvalues λ 0 , λ 1 of the Jacobian.
• Local geometric properties: Skewness can be measured as the ratio of the circumcircle to incircle radius, while regularity is inferred from element area variance.
Mesh quality can be quantified through:
• Scale: Element size, measured as λ 0 λ 1 , compared to a natural length scale.
• Skewness: The anisotropy of elements, given by λ 1 /λ 0 .
• Regularity: Consistency of adjacent elements, e.g., variance in element areas.
• Consistency: Stability of element shapes across the domain.
Aspect Ratio for our evaluation we use the aspect ratio of a triangular element, which is defined as the ratio of the longest edge l max to the shortest altitude h min :
AR = l max h min ,(27)
where h min is the shortest perpendicular distance from the opposite vertex to the longest edge. A higher aspect ratio indicates more elongated elements.
this section cite: []

Section: B. Notes on the use of Firedrake in G-Adaptivity
Firedrake (Ham et al., 2023) is a Python framework for the automatic solution of finite element problems. The central design idea based on composable abstractions, which allow the expression of the partial differential equation in weak form at a high level in Unified Form Language (UFL) (Alnaes et al., 2014). This abstraction is gradually lowered to generate C-kernels for matrix-assembly that can be executed in grid traversal with PyOP2 (Rathgeber et al., 2012). PETSc (Balay et al., 2019) provides a wide range of linear-and non-linear solvers for the resulting linear algebra problem. Firedrake supports a broad collection of finite element discretisations and dolfin-adjoint (Mitusch et al., 2019) allows the automatic construction of the adjoint problem for a given forward equation. The recently added interface to PyTorch (Bouziani & Ham, 2023) is crucial for the work in this paper.
this section cite: ['b7', 'b3', 'b51', 'b4', 'b45', 'b7']

Section: B.1. Additional details on implementation
Training the GNN requires computing the derivative of the loss function E(Z, U Z ) with respect to node coordinates Z. Since E(Z, U Z ) is a PDE-constrained functional, it is necessary to use adjoint models to compute these derivatives efficiently. The derivative and adjoint formulas depend on the loss function and its PDE constraints and automating their derivation is crucial to develop a general r-adaptivity methodology that can be trained seamlessly on different test cases. Firedrake is the perfect tool for this because it can derive adjoint models (Farrell et al., 2013;Mitusch et al., 2019) and automatically compute derivatives of E(Z, U Z ) with respect to node coordinates (Ham et al., 2019b). Deriving these formulas by hand is nontrivial, tedious, and error prone. Firedrake is fully integrated with PyTorch (Bouziani et al., 2024), and this is key to formulate hybrid FEM-torch architectures required to train the GNN. Implementing our approach in Firedrake required minimal adaptations: the GNN model must conform to the Firedrake external operator API, and a term must be added to the derivatives with respect to node coordinates when E(Z, U Z ) comprises a finite element solution computed on a finer mesh.
As an example consider the shape derivative dJ(Z, U Z )[T ] of the functional J(Z, U Z ) = ||U Z || 2 L2(Ω) , which is a simplified version of E(Z, U Z ) in (3). The constraint on U Z is given by the simplest testcase: the weak Poisson equation in Appendix C.1 with f = 4. With Firedrake and PyAdjoint, we can compute dJ(Z, U Z )[T ] as follows: mesh = UnitSquareMesh(3, 3) continue_annotation() Q = mesh.coordinates.function_space() T = Function(mesh.coordinates.function_space()) mesh.coordinates.assign(mesh.coordinates + T) V = FunctionSpace(mesh, \CG", 1) u = Function(V) v = TestFunction(V) solve ((dot(grad(u)
,grad(v))-4 * v) * dx==0, u, bcs=DirichletBC(V, 0, \on_boundary")) J = assemble(u ** 2 * dx) Jred = ReducedFunctional(J, Control(T)) Jred.derivative()
Crucially, this only requires the implementation of the forward constraint equation in Appendix C.1. On the other hand, a tedious manual derivation of
dJ(Z, U Z )[T ] leads to dJ(Z, U Z )[T ] = Ω (U 2 Z + ∇U Z • ∇p -4p)∇ • T -∇U Z (DT + DT ⊤
)∇p dx with p being the (weak) solution of the adjoint equation ∆p = 2U Z . These formulae are problem dependent and will be significantly more complicated for other PDE constraints. For test cases such as the Navier Stokes equations in Appendix C.3 this approach quickly becomes intractable, as highlighted in (Ham et al., 2019b(Ham et al., , p. 1818)).
In contrast, adapting the code above to the problems described in Appendix C.2 & C.3 requires only minor changes.
this section cite: ['b21', 'b45', 'b8']

Section: C. Mathematical description of the numerical experiments C.1. Poisson's equation
Poisson's equation -∇ 2 u = f (z) is solved using the Finite Element method in the two-dimensional convex domain z ∈ Ω ⊂ R 2 . We use the weak formulation (2) and seek piecewise linear functions u ∈ S Z with Dirichlet Boundary
Step 3: Update velocity Find u n+1 ∈ Q d-1 Z such that: Ω u n+1 • v dx = Ω u * • v dx -∆t Ω ∇(p n+1 -p n ) • v dx.(21)
for all piecewise quadratic vector-valued test functions v ∈ Q d-1 Z . The weak problem in ( 21) is solved with a conjugate gradient iteration preconditioned with SOR.
D. Further details of numerical experiments D.1. Model and data hyperparameters Table 4 shows for each PDE and geometry the number of train and test set samples as will as the resolution or node count for the train, test dataset and evaluation mesh.
PDE Poisson Burgers Navier-Stokes Domain Square Polygonal Square Cylinder Train/Test Samples 100/100 100/100 100/100 25/50 Train Resolution [15x15, 20x20] 114 nodes [15x15, 20x20] 201 nodes Test Resolution [12x12,...,23x23] 114 nodes [12x12,...,23x23] 201 nodes Eval Resolution 100x100 228 nodes 100x100 402 nodes
Table 4. Summary of PDE problem setups, including domains, sample sizes, and training/testing/evaluation resolutions.
this section cite: []

Section: D.4. Further experiments on non-convex domains
While section 5.4 already contains an example of a non-convex domain we provide further evidence that our method extends to this case using domain data from UM2N (Zhang et al., 2024). In particular we conducted experiments similar to the setup of 5.2 on five non-convex domains, cf. Figure 12. On each domain we solve Poisson's equation for randomly sampled Gaussian solutions with 100 training datapoints and 100 unseen test datapoints.
The results (error reduction scores are listed in in Table 5 and the full results can be seen in Figure 12) confirm that our method performs robustly on non-convex geometries, achieving significantly greater error reduction than baselines and generating regular non-tangled meshes on all tested domains, succeeding even when some other approaches fail. Note that the UM2N results reported below were obtained using the pretrained model from the UM2N repository, since the MA meshes obtained using (Wallwork et al., 2024) were unsuitable for direct training in these cases. * Monge-Ampère solvers in general struggle with non-convex domains. † Since the MA data available is not suitable for accurate training we use the pretrained model from (Zhang et al., 2024) for our evaluation.
this section cite: ['b58', 'b56', 'b58']

Section: D.5. Model hyper-parameter sensitivity analysis
We have performed extensive sensitivity studies and found that our approach is robust to the particular choice of hyperparameters for the diffformer blocks. We used N b = 4 blocks and rollout using explicit Euler time integration for 32 timesteps with a step size of 0.1. It should be noted that the hyperparameters were identical in all experiments performed in the paper and did not require finetuning to the specific problem. Tables 6 and 7 show the sensitivity of the model to the diffusion parameters in terms of error reduction and inference time. We investigated the sensitivity of G-Adaptivity to the weighting of the equidistribution regularizer in Section 4.4. We found Definition F.3 (Jacobian of mesh Deformation Map). Given the mesh deformation model M θ : (X, A) → X , the Jacobian J of the transformation is given by:
J = ∇M(X),
where ∇M(X) is the local derivative of the deformation map.
Definition F.4. (Mesh Tangling). We say that a physical mesh is tangled if at least one simplex in the triangulation has a negative determinant in its Jacobian matrix, i.e., det(J i ) ≤ 0, for some i, where J i is the Jacobian matrix of the affine transformation mapping the reference element to the physical element in the mesh. Equivalently, the mesh is untangled if all eigenvalues of the Hessian of the transformation function, or its discrete counterpart given by the graph Laplacian, remain positive.
this section cite: []

Section: Proof of Mesh Tangling Prevention
We prove that a Laplacian GNN-based mesh adaptation scheme prevents tangling, given a sufficiently small time step. The argument follows from the positivity of the determinant of the mesh deformation Jacobian, which is preserved due to the eigen-structure of the graph Laplacian.
this section cite: []

Section: F.1. Continuous-Time Evolution
The evolution of node positions follows the Laplacian-based update:
dX dt = (A -I)X = -∆X.
where ∆ is the weighted random-walk graph Laplacian. As A is frozen over every diffusion block, the solution of this ordinary differential equation is:
X(t) = e -t∆ X(0),(25)
implying the determinant of the transformation Jacobian satisfies
J(t) = det(e -t∆ )J(0) = i e -tλi J(0) = e -t tr(∆) J(0).(26)
Since tr(∆) = N x × d ≥ 0, we have J(t) > 0 for all t ≥ 0, ensuring that no elements invert.
this section cite: []

Section: F.2. Time Step Constraints for Mesh Preservation
The discrete update for the mesh is X k+1 = (I -dt∆)X k , propagating the determinant as J k+1 = det(I -dt∆)J k . To prevent inversion, we require det(I -dt∆) > 0.
this section cite: []

Section: Theorem F.5 (Time Step Condition for Mesh Preservation).
Given the discrete update X k+1 = (I -dt∆)X k , the mesh remains untangled if dt < 1 2 .
Proof. The determinant of the deformation Jacobian propagates as det J k+1 = det(I -dt∆) det J k . To ensure det J k+1 > 0, we require det(I -dt∆) > 0. The eigenvalues of I -dt∆ are µ i = 1dtλ i so the determinant condition reduces to i (1dtλ i ) > 0.
Noting that A has positive entries with row sum equal to 1, it follows by Gershgorin's theorem that the eigenvalues of A -I are contained in the Gershgorin circle |λ i -1| < 1. Seeing as the coefficients of A -I are real-valued the eigenvalues of A -I are either real-valued or come in complex conjugate pairs. If λ i is real-valued the contribution to the above determinant is 1 -
dtλ i > 1 -2dt > 0 if dt > 1/2.
If Imλ i ̸ = 0 then λ i is also an eigenvalue and the contribution to the determinant is (1
-dtλ i )(1 -dtλ i ) = |1 -dtReλ i | 2 + |Imλ i | 2 > 0.
Hence we obtain det(I -dt∆) > 0 if dt < 1 2 .
this section cite: []

Section: References
Ref_id:b0 Title: The deal.ii library, version 9.6 Year: (2024)
Ref_id:b1 Title: A posteriori error estimation in finite element analysis Year: (1997-03)
Ref_id:b2 Title: Graph Element Networks: Adaptive, structured computation and memory Year: (2019-05)
Ref_id:b3 Title: Unified form language: A domain-specific language for weak formulations of partial differential equations Year: (2014)
Ref_id:b4 Title: Petsc users manual Year: (2019)
Ref_id:b5 Title: A generic grid interface for parallel and adaptive scientific computing. part ii: implementation and tests in dune Year: (2008)
Ref_id:b6 Title: Relational inductive biases, deep learning, and graph networks Year: (2018)
Ref_id:b7 Title: Physics-driven machine learning models coupling PyTorch and Firedrake Year: (2023)
Ref_id:b8 Title: Differentiable programming across the PDE and Machine Learning barrier Year: (2024)
Ref_id:b9 Title: Message Passing Neural PDE Solvers Year: (2022-02)
Ref_id:b10 Title: Geometric deep learning: Going beyond euclidean data Year: (2017)
Ref_id:b11 Title: Monge-Ampére based moving mesh methods for numerical weather prediction, with applications to the Eady problem Year: (2013-03)
Ref_id:b12 Title: Adaptivity with moving grids Year: (2009-05)
Ref_id:b13 Title: p4est: Scalable algorithms for parallel adaptive mesh refinement on forests of octrees Year: (2011)
Ref_id:b14 Title: Beltrami Flow and Neural Diffusion on Graphs Year: (2021)
Ref_id:b15 Title: GRAND: Graph Neural Diffusion Year: (2021-07)
Ref_id:b16 Title: The numerical solution of the navier-stokes equations for an incompressible fluid Year: (1967)
Ref_id:b17 Title: Numerical solution of the navier-stokes equations Year: (1968)
Ref_id:b18 Title: Compatible finite element methods for geophysical fluid dynamics Year: (2023-05)
Ref_id:b19 Title: Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering Year: (2016)
Ref_id:b20 Title: The Deep Ritz Method: A Deep Learning-Based Numerical Algorithm for Solving Variational Problems Year: (2018-03)
Ref_id:b21 Title: Automated derivation of the adjoint of high-level transient finite element programs Year: (2013)
Ref_id:b22 Title: Deep reinforcement learning for adaptive mesh refinement Year: (2023-10)
Ref_id:b23 Title: Swarm reinforcement learning for adaptive mesh refinement Year: (2023)
Ref_id:b24 Title: Neural Message Passing for Quantum Chemistry Year: (2017-07)
Ref_id:b25 Title: Understanding convolution on graphs via energies Year: (2023-06)
Ref_id:b26 Title: Can Physics-Informed Neural Networks beat the Finite Element Method? Year: (2023)
Ref_id:b27 Title: Solving Ordinary Differential Equations II Year: (1996)
Ref_id:b28 Title: Automated shape differentiation in the unified form language. Structural and Multidisciplinary Optimization Year: (2019)
Ref_id:b29 Title: Automated shape differentiation in the Unified Form Language. Structural and multidisciplinary optimization Year: (2019)
Ref_id:b30 Title:  Year: ()
Ref_id:b31 Title: Solving high-dimensional partial differential equations using deep learning Year: (2018)
Ref_id:b32 Title: Better Neural PDE Ssolvers Through Data-Free Mesh Movers. The Twelfth International Conference on Learning Representations Year: (2024)
Ref_id:b33 Title: Adaptive Moving Mesh Methods Year: (2011-01)
Ref_id:b34 Title: Highly accurate protein structure prediction with AlphaFold Year: (2021-08)
Ref_id:b35 Title: Semi-Supervised Classification with Graph Convolutional Networks Year: (2022-07)
Ref_id:b36 Title: Learning skillful medium-range global weather forecasting Year: (2023)
Ref_id:b37 Title: Multipole Graph Neural Operator for Parametric Partial Differential Equations Year: (2020)
Ref_id:b38 Title: Fourier Neural Operator for Parametric Partial Differential Equations Year: (2020-09)
Ref_id:b39 Title: Thirty-Seventh Conference on Neural Information Processing Systems Year: (2023-11)
Ref_id:b40 Title: Learning the Dynamics of Physical Systems from Sparse Observations with Finite Element Networks Year: (2022-03)
Ref_id:b41 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021)
Ref_id:b42 Title: Automated solution of differential equations by the finite element method: The FEniCS book Year: (2012)
Ref_id:b43 Title: Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators Year: (2021-03)
Ref_id:b44 Title: Optimal-Transport-Based Mesh Adaptivity on the Plane and Sphere Using Finite Elements Year: (2018)
Ref_id:b45 Title: dolfin-adjoint 2018.1: automated adjoints for fenics and firedrake Year: (2019)
Ref_id:b46 Title: Learning Mesh-Based Simulation with Graph Networks Year: (2023-02)
Ref_id:b47 Title: A numerical study of some hessian recovery techniques on isotropic and anisotropic meshes Year: (2011)
Ref_id:b48 Title: A new implementation of the adaptive mesh transform in the Met Office 3D-Var system Year: (2012)
Ref_id:b49 Title: Forward-Backward Stochastic Neural Networks: Deep Learning of High-dimensional Partial Differential Equations Year: (2018-04)
Ref_id:b50 Title: Physicsinformed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations Year: (2019-02)
Ref_id:b51 Title: Pyop2: A high-level framework for performance-portable simulations on unstructured meshes Year: (2012)
Ref_id:b52 Title: M2N: Mesh Movement Networks for PDE Solvers Year: (2022-05)
Ref_id:b53 Title: Enhancing data representation in forging processes: Investigating discretization and R-adaptivity strategies with Proper Orthogonal Decomposition reduction Year: (2024)
Ref_id:b54 Title: Numerical comparison of some hessian recovery techniques Year: (2007)
Ref_id:b55 Title: Graph Attention Networks Year: (2018-02)
Ref_id:b56 Title:  Year: (2024-05)
Ref_id:b57 Title: MMPDE-Net and Moving Sampling Physics-informed Neural Networks Based On Moving Mesh Method Year: (2023-11)
Ref_id:b58 Title:  Year: (2024)
