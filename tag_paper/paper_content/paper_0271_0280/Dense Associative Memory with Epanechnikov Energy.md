Title: Dense Associative Memory with Epanechnikov Energy
Abstract: We propose a novel energy function for Dense Associative Memory (DenseAM) networks, the log-sum-ReLU (LSR), inspired by optimal kernel density estimation. Unlike the common log-sum-exponential (LSE) function, LSR is based on the Epanechnikov kernel and enables exact memory retrieval with exponential capacity without requiring exponential separation functions. Moreover, it introduces abundant additional emergent local minima while preserving perfect pattern recovery -a characteristic previously unseen in DenseAM literature. Empirical results show that LSR energy has significantly more local minima (memories) that have comparable log-likelihood to LSE-based models. Analysis of LSR's emergent memories on image datasets reveals a degree of creativity and novelty, hinting at this method's potential for both large-scale memory storage and generative tasks.

Section: Associative Memories and Energy Landscapes
Energy-based Associative Memory networks or AMs are models parameterized with M "memories" in d dimensions, Ξ = {ξ µ ∈ R d , µ ∈ M }. A popular class of models from this family can be described by an energy function defined on the state vector x ∈ X ⊆ R d :
E β (x; Ξ) = -Q M µ=1 F (βS (g(x), ξ µ )) ,(1)
where g : R d → R d is a vector operation (such as binarization, (layer) normalization), S : R d × R d → R is a similarity function (e.g., dot-product, negative Euclidean distance), β > 0 denotes the inverse temperature, F : R → R is a rapidly growing separation function (power, exponential) and Q is a monotonic scaling function (logarithm, linear) [1,2,3]. With g as the sign-function, ξ µ ∈ {-1, +1} d , S(x, x ′ ) = ⟨x, x ′ ⟩ and F as the quadratic function, and Q as a linear function, we recover the classical Hopfield model [4]. The output of an associative memory (AM) corresponds to one of the local minima of its energy function. A memory ξ µ is said to be retrieved if x ≈ ξ µ corresponds to such a local minimum, with exact retrieval occurring when x = ξ µ . The memory capacity of the AM is defined as the maximum number M ⋆ of correctly retrieved memories. For classical AMs, the capacity scales as M ⋆ = O(d). With the introduction of power-law separation functions -that is, F (x) = x p for p > 2 -modern Dense Associative Memories (DenseAMs) achieve a significantly higher capacity of M ⋆ = O(d p ) [1,5].
The use of an exponential separation function combined with a logarithmic scaling function -F (x) = exp(x) and Q(x) = log x -leads to the widely studied log-sum-exp (LSE) energy function [6,7,8], yielding exponential memory capacity M ⋆ ∼ exp(d) [9]. Hierarchical organization of memories has also been explored in Krotov [10] and Hoover et al. [11]. Given that the gradient 39th Conference on Neural Information Processing Systems (NeurIPS 2025). of the LSE energy corresponds to a softmax over all stored patterns, recent works have proposed sparsified variants to improve scalability. In particular, Hu et al. [12] and dos Santos et al. [13] consider sparsified softmax-based gradients, effectively projecting the full gradient onto a reduced support. 1 Alternatively, Wu et al. [14] propose learning new representations for the memories to increase capacity, while continuing to use the LSE energy in the transformed representation space.
In this work, we consider the following motivating question: can we achieve simultaneous perfect memorization and generalization in associative memory models? While the exponential separation function enables DenseAMs to achieve high memory capacity, capacity alone is not the only desideratum. In standard supervised learning, it was long believed that exactly interpolating or memorizing the training data -achieving zero training loss -would harm generalization. However, recent advances, particularly in deep learning, have challenged this belief: models that perfectly fit the training data can still generalize well. Although this phenomenon gained prominence with deep networks, it has earlier roots in kernel methods and boosting [15,16,17].
An analogous question arises in the context of associative memory (AM) models. Traditionally, AMs focus on storing a fixed set of patterns. But from a broader machine learning perspective, the goal extends beyond memorization to include the generation of new, meaningful patterns. Prior work using LSE-type energy functions has shown that generating such novel patterns typically requires sacrificing perfect recall of the original patterns. This trade-off highlights a core tension between memorization and generalization. To address this, we explore alternative separation functions that can preserve exact memorization while enabling the emergence of new patterns -pushing toward models that truly unify memory and generalization.
Our approach is also motivated by the well-established connection between the energy and probability density function. An energy function E : R d → R induces a probability density function p : R d → R ≥0 with p(x) = exp[-E(x)]/ z exp[-E(z)]dz. Conversely, given a density p, we have an energy E(x) ∝ -log p(x), the negative log-likelihood. Minimizing the energy corresponds to maximizing the log-likelihood (with respect to the corresponding density). Based on this connection, with Q(•) = log(•), the exp[-E β (x; Ξ)] = µ F (βS(x, ξ µ )) in eq. ( 1) (assuming g is identity) is the corresponding (unnormalized) density at x. Assuming that the memories ξ µ ∼ p are sampled from an unknown ground truth density p, the exp[-E β (x; Ξ)] is an unnormalized kernel density estimate or KDE of p at x with the kernel F and bandwidth 1/β [18]. Thus, the LSE energy with F (x) = exp(x) and S(x, x ′ ) = -1 /2∥x -x ′ ∥ 2 corresponds to the KDE of p with the Gaussian kernel.
KDE is well studied in nonparametric statistics [18,19], and various forms of kernels have been explored. The quality of the estimates are well characterized in terms of properties on the kernels; we will elaborate on this in the sequel. While the Gaussian kernel is extremely popular for KDE (much like LSE in AM literature), there are various other kernels which have better estimation abilities than the Gaussian kernel. Among the commonly used kernels, the Epanechnikov kernel has the most favorable estimation quality (see section 2). In our notation, this corresponds to a kernel F (x) = max(1 + x, 0) = ReLU (1 + x), a shifted ReLU operation (again with S(x, x ′ ) = -1 /2∥x -x ′ ∥ 2 ). This results in a novel energy function that we name log-sum-ReLU or LSR (see eq. ( 3)). Surprisingly, we show that this energy function is capable of both exactly memorizing all M original patterns (with M ∼ exp(d)) and simultaneously generating new, meaningful emergent memories (see Definition 2). This defies the conventional tradeoff seen in prior AM models -where improving generalization (ability to create emergent local minima) typically requires compromising exact memorization -and reveals that precise memory storage and creative pattern generation are not inherently at odds with one another. To summarize, we make the following contributions in this work:
• Novel ReLU-based energy function with exponential memory capacity. We propose a LSR energy function for DenseAM using the popular ReLU activation, built upon the connection between energy functions and densities. In Theorems 1 and 2 respectively, we demonstrate exact retrieval and exponential memory capacity of LSR energy, without the use of exp(•) separation function.
• Simultaneous memorization and emergence. We show that this LSR energy has a unique property of simultaneously being able to exactly retrieve all original memories (training data points) while also creating many emergent memories (additional local minima). The total number of memories of LSR can exceed the number of stored patterns, a property absent with LSE (see Proposition 1). When applied to images, LSR can generate novel and seemingly creative memories that are not present in the training dataset.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b0', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b17', 'b18']

Section: Kernel Density Estimation and the Choice of Kernels
We now provide a brief overview of Kernel Density Estimation (KDE) considering the univariate setting for simplicity; similar conclusions also hold in higher dimensions. Given a sample Ξ = {ξ µ ∈ R, µ ∈ M } drawn from an unknown density f , the KDE is defined as
fh (ξ) = (M h) -1 M µ=1 K ξ-ξµ h
, where K(•) is the kernel function and h > 0 is the bandwidth parameter. The kernel function is assumed to satisfy: (i) symmetry (i.e., K(-x) = K(x), for all x ∈ R), (ii) positivity (i.e., K(x) ≥ 0, for all x ∈ R) and (iii) normalization (i.e., x K(x) dx = 1).
Note that for the purpose of KDE, the scale of the kernel function is not unique. That is, for a given K(•), we can define K(•) = b -1 K(•/b), for some b > 0. Then, one obtains the same KDE by rescaling the choice of h. Hence, the shape of the kernel function plays a more important role in determining the choice of the kernel. We now introduce two parameters associated with the kernel, µ K := x x 2 K(x) dx and σ K := x K 2 (x) dx that correspond to the scale and regularity of the kernel. We will discuss below how the generalization error of KDE depends on the aforementioned parameters.
The generalization error of fh (ξ) is measured by the Mean Integrated Squared Error (MISE), given by MISE(h) = E ξ ( fh (ξ) -f (ξ)) 2 dξ . Assuming that the ground-truth density f (ξ) is twice continuously differentiable, a second-order Taylor expansion gives the leading terms of the MISE(h), which decomposes into squared bias and variance terms:
MISE(h) ≈ µ 2 K 4 h 4 ξ |f ′′ (ξ)| 2 dξ + σ K M h
; see Wand and Jones [18, Section 2.5] for details. This result shows that reducing h decreases bias but increases variance, while increasing h smooths the estimate but introduces bias, highlighting the bias-variance trade-off. The optimal mean-square is obtained by minimizing MISE(h) with respect to h. We thus obtain the optimal choice of h and the optimal generalization accuracy as
h * := σ K M µ 2 K 4 ξ |f ′′ (ξ)| 2 dξ 1/5
and MISE(h * ) ≈ 5 4
√ µ K σ K ξ |f ′′ (ξ)| 2 dξ M 4/5 ,(2)
respectively. From this, we see that the choice of the kernel K in the KDE, controls the generalization error via the term √ µ K σ K .
Thus, a natural question is to find the choice of kernel K(•) that results in the minimum MISE(h * ).
As discussed above, the scale of the kernel function is non-unique. Hence, the problem boils down to minimizing σ K (which is regularity parameter of the kernel, determining the shape), subjected to µ K = 1 (without loss of generality), over the class of normalized, symmetric, and positive kernels.
This problem is well-studied (see, for example, [20], [21], Wand and Jones [18, Section 2.7]), and, as it turns out, the Epanechnikov kernel K epan (x) = max{1 -x 2 , 0} = ReLU 1 -x 2 achieves the optimal generalization error. The quantity, Eff(K) := σ K /σ Kepan is hence referred to as the efficiency of any kernel with respect to the Epanechnikov kernel. A description of choices for kernel functions and their efficiency relative to the Epanechnikov kernel is provided in Appendix C.
The number of modes of the KDE has also been examined in the literature, mostly when the target is unimodal. Assuming that the target is unimodal, a direct consequence of Mammen [22, Theorem 1] on the number of modes of the KDE when d = 1 (also see Geshkovski et al. [23, Theorem 1.1]) is that the number of modes of KDE with a Gaussian kernel with bandwidth h is Θ(1/ √ h); see Geshkovski et al. [23, Section 1.2] for extensions to dimension d > 1.
this section cite: ['b19', 'b20']

Section: A New Energy Function with Emergent Memory Capabilities
So far, we have seen the relationship between the LSE energy and the KDE, i.e., exp[-E β (x; Ξ)] is an unnormalized kernel density estimate with the Gaussian kernel and the bandwidth 1/β, and the optimality of using the Epanechnikov kernel in KDE. Given these observations, we will explore the use of the corresponding shifted-ReLU separation function ReLU (1 + x) in the energy function instead of the widely used exponentiation. Before we state the precise energy functions, we compare and contrast the shapes of these separation functions F (βx) in fig. 2 for varying values of the inverse temperature β. Note that, as the β increases, both these separation functions decay faster. However, as expected, the shifted-ReLU separation linearly decays and then zeroes out.
this section cite: []

Section: Recall that LSE ENERGY is given by E
LSE β (x; Ξ) = -1 β log M µ=1 exp(-β 2 ∥x -ξ µ ∥ 2 ).
Based on the discussion on separation functions, our proposed LSR ENERGY (which we also refer to as Epanechnikov energy) is given by
E LSR β (x; Ξ) = - 1 β log ϵ + M µ=1 ReLU 1 - β 2 ∥x -ξ µ ∥ 2 ,(3)
where ∥•∥ describes the Euclidean norm and β is an inverse temperature.
4 2 0 x = S(x, x ) 0.0 0.5 1.0 F( x) F( x) = exp( x) 4 2 0 x = S(x, x ) F( x) = ReLU(1 + x) =1.0 =0.5 =0.2
′ ) = -1 /2∥x -x ′ ∥ 2 .
The factor ϵ ≥ 0 in the LSR energy is a small nonnegative constant, where an ϵ > 0 ensures that every point in the space has finite (albeit extremely large O(log(1/ϵ))) energy for all values of β. Indeed, with ϵ = 0, defining S µ ≜ {x ∈ X : ∥x -ξ µ ∥ ≤ 2 /β}, it is easy to see that ∀x ∈ X \ ∪ M µ=1 S µ , E LSR β (x) = ∞. This is a result of the finite-ness of the ReLU separation function. Regions of infinite energy implies zero probability density, which matches the finite support of the density estimate with the Epanechnikov kernel. Based on the introduced LSR energy, we next highlight the following favourable properties; see appendix E for the proofs and technical details. Theorem 1. Let r = min µ,ν∈ M ,µ̸ =ν ∥ξ µ -ξ ν ∥ be the minimum Euclidean distance between any two memories. Let S µ (∆) = {x ∈ X : ∥x -ξ µ ∥ ≤ ∆} be a basin around the µ th memory for some basin radius ∆ ∈ (0, r). Then, with β = 2/(r -∆) 2 , for any µ ∈ M and any input x ∈ S µ (∆), the output of the DenseAM via LSR energy gradient descent is exactly ξ µ , implying that all memories ξ µ , µ ∈ M are retrievable. Furthermore, if the learning rate of the energy gradient descent is set appropriately, then for any µ ∈ M and any x ∈ S µ (∆), the memory is exactly retrieved with a single energy gradient descent step (single step retrieval).
The above result states that, given a set of memories, and an appropriately selected β, there is a distinct basin of attraction S µ (∆) around each memory ξ µ , and any input x from within that basin exactly retrieves the memory as the output of the DenseAM. Remark 1. For a finite but appropriately large β, the LSR energy gradient ∇E LSR β (ξ µ ; Ξ) at any memory ξ µ is exactly zero, implying exact retrieval of the memory. The LSE energy gradient ∇E LSE β (ξ µ ; Ξ) is only approximately zero, and the retrieved point is approximately equal to an original memory [7,Theorem 3]. However, if β = ∞ then the LSE energy gradient is exactly zero at the memory.
The striking phenomenon that we observe with LSR energy is that the DenseAM can simultaneously create local energy minima around the original memories as well as additional local minima around points that are not part of the set of original memories; see fig. 1. We formalize this concept below through the notion of global emergence.
Definition 1 (Novel local minima). Consider a DenseAM parameterized with M memories Ξ = {ξ 1 , . . . , ξ M }, ξ µ ∈ X , and equipped with an energy function E β (x; Ξ) at any state x ∈ X for a specific inverse temperature β > 0. For some ε > 0, we define M ε as the (possibly empty) set of novel local minima ξ ∈ X such that ∀ ξ ∈ M ε , (a) ξ is a local energy minimum with ∇E β ( ξ; Ξ) = 0 and ∇ 2 E β ( ξ; Ξ) ≻ 0, (b) ξ is novel with respect to the original memories, that is, min µ∈ M ξξ µ ≥ ε.
Definition 2 (Global emergence). For the DenseAM in Definition 1 and for some ε > 0, let M ε be the (possibly empty) set of novel local minima. For some β ⊂ (0, ∞), we claim that this system exhibits ε-global emergence if (i) each original memory ξ µ , µ ∈ M is a local energy minimum with ∇E β (ξ µ ; Ξ) = 0 and ∇ 2 E β (ξ µ ; Ξ) ≻ 0 (positive definite), and (ii) the set M ε is non-empty. We term M ε as the set of ε-globally emergent memories.
The notion of ε-global emergence specifically refers to those new patterns that arise after all original memories have been exactly stored. Definition 2 characterizes emergence as a property of the global energy function at a specific inverse temperature, requiring simultaneous exact recovery of all original memories and the presence of at least one novel local minimum (parameterized with ε). It is instructive to start by understanding the above definition for DenseAMs equipped with LSE energy. According to Ramsauer et al. [7], any point x * such that ∇E LSE β (x * ; Ξ) = 0 is defined via the softmax corresponding to the transformer attention as follows: x * = M µ=1 softmax(β(x * ) ⊤ ξ µ )ξ µ , and the softmax can be highly peaked if all {ξ µ } M µ=1 are well separated and x * is near a stored pattern ξ µ . If no stored pattern ξ µ is well separated from the others, then x * is close to a global fixed point, which is the arithmetic mean of all the stored patterns. Based on this, we can make the following observations:
• Case I: All LSE memories are novel. With a large enough but finite β, there is a minimum close to each of the original memories. However, each of these local minima will be considered novel local minima as these are distinct from the original memories, thus condition (ii) in Definition 2 will be satisfied. However, then the condition (i) in Definition 2 would not be satisfied.
• Case II: No LSE memories are novel. If we do consider the case β = ∞, then the original memories would exactly be the local energy minima, and condition (i) will be satisfied. But then the set of novel local minima M ε for a strictly positive ε would be empty, violating condition (ii).
• Case III: Novel LSE memories form only when basins merge. For a moderate β, LSE can form novel local minima by merging the basins of attractions of the original memories, thereby giving us a non-empty M ε and satisfying condition (ii) in Definition 2. However, condition (i) will be violated as the memories whose basins are merged would no longer be local minima.
Thus, we can make this more formal in the following: Proposition 1. Assume {ξ µ } M µ=1 are i.i.d. from any density fully supported on X . Note that they are linearly independent with probability 1, as otherwise they lie in a lower dimensional space. Then, for any β > 0, the LSE energy, E LSE β (•), does not satisfy the ε-global emergence in Definition 2.
One can argue that global emergence as in Definition 2 is too restrictive; we also want to characterize an individual local minimum as "emergent", or not. Thus we present a relaxed local notion of emergence in the following, noting that LSE does not satisfy this weaker form of emergence either: Definition 3 (Locally emergent memory). Consider the DenseAM in Definition 1 with a non-empty set of novel local minima M ε for a ε > 0. For any ξ ∈ M ε , let S( ξ) ⊆ Ξ be the minimal nonempty subset of Ξ such that, for each ξ µ ∈ S( ξ), ξ is no longer a local minimum of the energy E β (•, Ξ \ {ξ µ }) that excludes the memory. Then we define ξ ∈ M ε as a ε-locally emergent memory if there is some original memory ξ µ ∈ S( ξ) which still is a local minimum of the energy
E β (•; Ξ). If every original memory ξ µ ∈ S( ξ) is a local minimum of E β (•; Ξ), we call ξ ∈ M ε a ε-local strongly emergent memory.
While Definition 2 discusses emergence at a global energy level, Definition 3 characterizes emergence locally for each of the novel local minima. This distinction is important as we see emergence as a general property of the energy function that can be driven by a subset of the memories. Definition 2 requires all original memories to be retrievable, while in Definition 3 we allow for emergence due to the interaction of stored patterns in a system even if not all original memories are retrievable, so long as a critical subset of the original memories are. Global emergence implies the existence of at least one local strongly emergent memory; every globally emergent memory is a local strongly emergent one. However, the existence of a local strongly emergent memory does not imply global emergence. As with global emergence, we can see that a DenseAM with LSE energy does not also have locally emergent memories. First note that, for a finite β, all local minima are novel local minima, and the minimal set S( ξ) for a novel local minimum ξ is the whole set Ξ given the infinite support of the exponential function, with none of them being a local minimum. For β = ∞, the set of novel local minima M ε is empty. So in both cases, the required conditions are not satisfied and there are no locally emergent memories with LSE energy.
Note that these novel local minima are different from the well-studied spurious memories or parasitic memories [24,25]. In classical AMs, spurious memories start appearing when the AM is packed with memories beyond its memory capacity. In contrast, the appearance of emergence (novel local minima) does not seem to be related to whether the DenseAM is over or under capacity -as we show in fig. 1, a locally emergent memory can appear even with just 2 stored patterns of any dimension. Spin-glass states [26] do not occur in either the LSE or LSR energy due to our use of Euclidean similarity over the dot product in both energies.
The next result provides explicit characterization of the form of novel memories in LSR energy.
this section cite: ['b6', 'b6', 'b23', 'b24', 'b25']

Section: Proposition 2. Consider the LSR energy in eq.
(3). For any x ∈ interior(X ), letting B(x) ≜ {µ ∈ M : ∥x -ξ µ ∥ ≤ 2/β}, there is a local minima of the LSR energy which is given by
1 |B(x)| µ∈B(x) ξ µ .
Note that when |B(x)| = 1, the local minima in Proposition 2 is exactly the stored memory {ξ µ } M µ=1 . With |B(x)| > 1, it is not equal to any of the original memories {ξ µ , µ ∈ M } (with probability 1). The region {x ∈ X : |B(x)| > 1} ⊂ X is precisely characterized as ∪ µ∈ M S µ \ ∪ µ∈ M S µ (∆) where S µ is the region of finite energy around the µ th memory and S µ (∆) (defined in Theorem 1) is the distinct attracting basin for the µ th memory. The following theorem shows that this LSR based DenseAM is capable of simultaneously retrieving all (up to exponentially many) memories while also creating many novel local minima, and quantifies this phenomenon precisely. Theorem 2. Consider a DenseAM parameterized with M memories Ξ = {ξ 1 , . . . , ξ M } sampled uniformly from X with vol(X ) = V < ∞ and the LSR energy
E LSR β (•; Ξ) defined in eq. (3). For each novel local minimum x * = 1 |B(x * )| µ∈B(x * ) ξ µ given in Proposition 2, define D max (x * ) := max µ ∥x * -ξ µ ∥, δ min (x * ) := min µ∈B(x * ) 2 β -∥x * -ξ µ ∥ 2 , γ min (x * ) = min ν / ∈B(x * ) ∥x * -ξ ν ∥ 2 - 2 β .
Then, for all β > 0 such that max x * δ min (x * ) > 0 and min x * γ min (x * ) > 0, there exists an ε := min
x * D 2 max (x * ) + min{δ min (x * ), γ min (x * )} -D max (x * ) > 0 such that E LSR β (•)
satisfies the ε-global emergence condition (Definition 2) with high probability, as:
(a) With probability at least δ ∈ (0, 1), and M = Θ √ 1 -δ exp(αd) for a positive α, all memories are retrievable as per Theorem 1 with the value of the minimum pairwise distance r = min µ,µ∈ M ,µ̸
=ν ∥ξ µ -ξ ν ∥ ≥ (V d /V ) -1/d e -2α and per-memory basin radius ∆ ∈ (0, (V d /V ) -1/d e -2α ) with a β ≤ 2/((V d /V ) -1/d e -2α -∆) 2 , where V d is the volume of the unit ball in R d . (b) For each novel local minimum x * , there exists a radius r * := D 2 max (x * ) + min{δ min (x * ), γ min (x * )} -D max (x * ) > 0,(4)
such that S x * (r * ) = {x ∈ X : ∥x -x * ∥ ≤ r * } forms a basin around the novel memory x * , and for any x ∈ S x * (r * ), the output of the DenseAM via energy gradient descent is exactly x * , implying the novel memories are retrievable.
Furthermore, with probability at least 1 -M -2 , the number of ε-globally emergent memories is
O exp M V d V 2 β d 2 log eV V d β 2 d 2 .(5)
In particular, for fixed β > 0 and d, the bound grows with M whenever β > 0 satisfies |B(x)| ∈ (1, M ] due to Proposition 2. Here, for any x ∈ X , a large β leads to a small |B(x)|, and |B(x)| > 1 is the required condition for a novel local minima; a small β leads to a large |B(x)|, and |B(x)| ≤ M stands for the case B(x) at most covers the entire domain X .
The above result demonstrates that with the LSR energy, it is possible to exactly memorize all original patterns (with high probability) and still generate new patterns -what we term emergent memories (Definition 2). This behavior is surprising in the same way interpolating models in deep learning generalize unexpectedly well: both challenge the classical bias-variance intuition [15,16].
While LSE-based models also produce novel memories, they typically do so at the expense of perfect recall of the original patterns. In Proposition 1 we show that LSE based DenseAMs do not have the global emergence property. This distinction highlights a key contribution of our work: new memory creation need not come at the cost of perfect memorization.
Next, we provide an exact order (i.e., upper and lower bounds) of the number of emergent memories under a grid design assumption.
Proposition 3. If {ξ µ } M µ=1 form a grid over X of equal size with Vol(X ) = V < ∞, the number of emergent memories is of order Θ M 1/d -λ 1/d + 1 d , where λ = Θ M V -1 (8/β) d 2
and for β > 0 such that 1 < λ ≤ M .
Note that we showed an explicit form of the emergent memories
x * = 1 B(x * ) µ∈B(x * ) ξ µ , where B(x * ) = µ : ∥x * -ξ µ ∥ < 2/β ⊂ {1, .
. . , M }. equation 5 in Theorem 2 is stated under the uniform sampling regime, and Proposition 3 is stated under a fixed grid setting. In general, the number of emergent memories varies according to the specific geometry of the stored patterns {ξ µ } M µ=1 , i.e., whether such a subset of {1, . . . , M } can be realized by a ball ∥x -ξ µ ∥ < 2/β . This can grow much faster than a linear order of M , and is naively bounded by 2 M .
this section cite: ['b14', 'b15']

Section: Experiments

this section cite: []

Section: Quantifying the scaling of emergent memories
How many local minima do we see in practice as we: (a) vary the number of stored patterns, (b) change the dimensionality of those patterns, and (c) vary the inverse temperature β? We observe that, at critical values of β, we can create orders of magnitude more emergent memories than stored patterns. These results are shown in fig. 3
this section cite: []

Section: (left).
To quantify the number of local minima induced by the LSR energy, we uniformly sample M patterns from the d-dimensional unit hypercube to serve as memories Ξ. We enumerate all possible local minima of the LSR energy by computing the centroid ξK := |K| -1 µ∈K ξ µ for every possible subset of stored patterns K ⊆ M (there are 2 M possible subsets if we allow for singleton sets). For each subset, we first check that its centroid is supported (i.e., that E LSR β ( ξK ; Ξ) < ∞ at ϵ = 0), and then declare that ξK is a local minimum of the LSR energy if ∇E LSR β ( ξK ; Ξ) < δ for small δ > 0. β values are varied across the "interesting" regime between fully overlapping support regions (a single local minimum in the unit hypercube) to fully disjoint support regions around each memory. See experimental details in appendix D.2.
Certain values of β yield particularly interesting behavior. For example, we observe that LSR can create orders of magnitude more emergent memories under ranges of β where: (i) a majority (>60%) of stored patterns are also recoverable, and (ii) around 20 percent of the unit hypercube is still supported. Note that in each experiment there are choices of β such that the LSR energy does not exhibit global emergence (Definition 2, i.e., at low β where novel memories are forming but not all stored patterns are yet retrievable). However, in these regions local emergence (Definition 3) of the novel memories still holds (see fig. 1 for intuition).
this section cite: []

Section: Generative quality of emergent memories
LSR memories are certainly more diverse than those of LSE, but do they represent more "meaningful" samples from a true, underlying density function p(x) (as measured by their log-likelihood)? The experimental setup is as follows: Let p(x) be a mixture of k Gaussians whose means µ i ∼ U([0, 1] d ) for i ∈ k are uniformly sampled from the d-dimensional unit hypercube with scalar (σ = 0.1) covariances such that p(x) = 1 k k i=1 N (x | µ i , σfoot_1 I d ). We sample M points {ξ 1 , . . . , ξ M }, ξ µ ∼ p(x) to serve as the stored patterns Ξ used to parameterize both the LSE and LSR energies from eq. ( 3). Define a thin support boundary induced by pattern ξ µ to be supp[ξ µ ; δ] = {x : 2β -1 -δ ≤ ∥x -ξ µ ∥ 2 < 2β -1 } for some small δ > 0. Then, for initial points x (0) n , n ∈ N sampled from the support boundary around each stored pattern, 2 LSE memories can be found using gradient descent
x (t) n = x (t-1) n -α∇E LSE β (x (t-1) n ; Ξ),(6)
until convergence to a memory x ⋆ n . We use algorithm 1 to efficiently find the LSR memory corresponding to each initial point. Thus we have N "samples" (memories) from both LSE and LSR on which we compare three metrics of interest in fig. 3 (right): 1. Average Log-Likelihood. Do LSR memories x ⋆ LSR have higher log p(x ⋆ LSR ) than LSE memories? 2. Number of Unique Samples. Does high log p(x ⋆ LSR ) occur alongside many emergent memories? All LSR memories + emergent memories emergent memories preserved memories / 40 preserved memories / 24 46 38 40 22 All LSE memories (total, at same β) 40 TinyImagenet [27] images are encoded into 256-dim latents using a pretrained VAE [28] and stored into an LSR-and LSE-energy using a carefully chosen β. Note that in this TinyImagenet example the LSR energy is, by definition, globally emergent since all stored patterns are recoverable, while the MNIST example is not. See experiment details in appendix D.4.
this section cite: ['b26', 'b27']

Section: Number of Original Memories Recoverable. Does high log p(x ⋆
LSR ) occur when alongside high numbers of preserved memories? How does this trend compare with LSE memory performance?
The results tell a consistent story. Despite LSE energy being a more natural choice to model the underlying Mixture of Gaussians' density p(x) (LSE has a Gaussian kernel that makes it ideal for the modeling task), LSR can match LSE in log-likelihood while simultaneously generating more diverse samples and preserving the stored patterns. See appendix D.3 for more experimental results and extended discussion.
this section cite: []

Section: Emergent memories in latent space
What do emergent memories look like when LSR is applied to real-world datasets? To study this behavior, we use a VAE to encode MNIST and TinyImagenet [27] images into latent vectors that serve as the stored patterns for LSR and LSE energies (see fig. 4). Using a carefully chosen β, we compute all memories (both preserved and novel) for each energy. The emergent memories of LSR in principle are simply the centroids of small subsets of the stored patterns, yet when decoded they appear as plausible and creative generations.
With the same β value, LSR generates an order of magnitude more total memories than LSE. While this choice of β is somewhat arbitrary and could be tuned separately for each energy, LSE would only ever be able to retrieve up to M memories (where M is the total number of stored patterns). See full experiment details in appendix D.4.
Emergent memories are mechanistically simple: they are simply the centroids of small subsets of the stored patterns. The semantic novelty of the emergent memories in Figure 4 occur because the latent space is structured to be semantically meaningful, where averaging two or more stored patterns produces seemingly novel semantics. We conduct a similar experiment in appendix D.5 where we ablate the VAE to show what emergent memories look like in pixel space.
this section cite: ['b26']

Section: Discussion
Emergent memories are a powerful tool for creating novel samples, but the "meaningfulness" of these samples is a nuanced question that depends heavily on the specific application domain and task requirements. For example, in Figure 3 (right) we show that the LSR energy approximates an unknown p.d.f. better than LSE's energy while simultaneously generating diverse samples. This represents a desirable behavior of emergence, since high log-likelihood samples from the LSE energy are quite homogeneous, causing 500 initial queries to converge to the same ∼10 memories. In this density estimation context, the emergent memories serve a clear functional purpose: they capture meaningful interpolations within the data distribution that improve generalization.
However, consider the novel memories from Tiny Imagenet in Figure 4. Though visually plausible, many of the emergent generations appear blurry and would be considered "undesirable" from the perspective of a high-fidelity image generation model. We discuss the limitations of emergent memories further in appendix A.
We note that there are potential parallels between emergent memories and "hallucinations" as observed in LLMs. We discuss the philosophical similarities between emergent memories and hallucinations in appendix B.1. It is also interesting to note that the LSR Energy presented in this work can have a feasible biological implementation using bipartite neurons. See further discussion in appendix B.2.
Finally, we reiterate that there are many choices for alternative kernels, and not just the Epanechnikov kernel. We discover that many kernels with compact support are capable of producing emergent memories and even manifolds, and the energy landscapes they produce can look quite different from each other. To this end, we show the "basin merging" behavior across different kernels in 1D in Figure 6 and we include an extended discussion on these other kernels in appendix C.
this section cite: []

Section: Conclusion
Our work introduces the LSR energy function which achieves the surprising combination of exact memorization of exponentially many patterns and the emergence of new, meaningful memories, thereby providing a powerful alternative to traditional AM formulations. The properties of the LSR energy define a novel class of emergent memories in Dense Associative Memory systems -a phenomenon not observed in any previous AM formulations. Unlike conventional models (e.g., LSEbased energies) where generalization (formation of local minima different from training data) typically comes at the cost of perfect memorization, LSR demonstrates that these objectives can coexist harmoniously in a single energy function. We additionally demonstrated that the diverse memories created by LSR achieve log-likelihood comparable to LSE when sampling from a true density function, while generating an order of magnitude more unique memories. Finally, we showed that when applied to latent representations of real-world image datasets, LSR's emergent memories represent plausible and creative generations.
this section cite: []

Section: A Limitations
To enable both emergence and exact memorization, the LSR energy studied in this work comes with certain limitations. Unlike the LSE energy whose gradient remains nonzero regardless of the query's distance from the stored patterns, the gradient of LSR energy vanishes exactly outside the support of the stored pattern. This makes gradient-based retrieval ineffective when the query lies far from any memorized pattern (though one can easily introduce a query-dependent temperature parameter β(x) that dynamically adjusts to ensure the query lies within at least one basin of attraction). Alternatively, one can create a "hybrid" energy function that combines the LSE and LSR energies, taking advantage of LSE's non-zero gradient everywhere outside LSR's support around the stored patterns. Such temperature-tuned DenseAMs and hybrid energies are of independent interest and we leave their systematic study to future work. Additionally, we want to emphasize that LSR can only create emergent memories precisely at the centroid of overlapping basins between stored patterns. This makes it possible to quickly and exactly retrieve memories (see Algorithm 3), but it also means that the "creativity" of this model is limited to a predictable subset of the convex hull of the stored patterns. The apparent novelty and creativity of Figure 4 is strongly aided by the semantic structure contained the VAEs' latent space.
this section cite: []

Section: B Extended Discussion

this section cite: []

Section: B.1 On the relationship between emergent memories and hallucination
In everyday use of LLMs, we call it a hallucination when the model confidently produces an incorrect fact, especially when we know the model is capable of producing the correct fact in other settings. To formalize this idea into the context of emergent memories, we must imagine a setting where the model has an explicit energy function and a known set of stored "facts." This is the setting of Associative Memory as studied in this work, where the outputs of our model are always energy minima (memories). A "hallucination" can occur when two or more stored facts interfere, leading the system to settle into an emergent minimum that looks meaningful (i.e., it lies near the "factual manifold" that generated our stored patterns) but does not correspond to any true, stored fact. Emergent memories are precisely these "interference minima" when they coexist with the original stored patterns. By definition, they are not stored facts, yet they can resemble meaningful combinations of them -making them the memory-system analog of hallucinations.
this section cite: []

Section: B.2 On a biological implementation of emergent memories
Dense AMs permit a standard mapping onto "biological" networks by introducing auxiliary hidden neurons following the method of [8]. Thus, the feature neurons in both LSE and LSR models can be augmented with auxiliary hidden neurons, resulting in a model with only pair-wise neuron interactions and bipartite connectivity between feature and hidden layers. This augmented model is more biological compared to the model considered in the main paper. The proposed model defined by the energy eq. ( 3) and the corresponding update rule eq. ( 7) can be obtained by integrating out these auxiliary hidden neurons, which means LSR energy will still have the same emergent behavior. The augmented model will still have some un-biological aspects because of the weight symmetry between forward and backward projections, which is necessary to ensure that the network has a global energy function.
this section cite: ['b7']

Section: C On Kernels
We show different kernels that are typically used for KDE and their efficiency relative to the Epanechnikov kernel in fig. 5. See the explanation on optimal kernel density estimation in section 2 for more details.
Though our main submission focuses on the Epanechnikov kernel, we find that the phenomenon of emergence (Definitions 2 and 3) is not limited to the Epanechnikov kernel. We state our findings for other popular kernels below and graphically in Figure 6. In summary, all compact support kernels are capable of producing emergent memories except for the TriWeight kernel (meanwhile, the uniform kernel is impractical for associative memory). Details below: points on the unit hypercube and computing the fraction of energies that are finite at ϵ = 0) as a fraction of the unit hypercube. At the smallest tested β, only a single energy minimum is induced at the centroid of all stored patterns with a region of support covering the whole unit hypercube. At the largest tested β, all original memories are recoverable and there are no spurious memories.
this section cite: []

Section: D.3 Details: Generative quality of memories D.3.1 Additional experiments
We ran the same experiment in section 4.2 under varying dimensions d = [8,16] and number of mixtures k = [5,10], averaging the results of each run across 5 different random seeds. The results for each of these experiments is shown below in figs. 7 and 8, where fig. 8 (left) is the same as reported in fig. 3 (right) in the main paper.
this section cite: ['b7', 'b15', 'b4', 'b9']

Section: D.3.2 Aligning β's across random seeds
We used the same setup for choosing an interesting range of β as we did in appendix D.2. However, over random seeds the value for r min can vary since it is dependent on the random choice of stored patterns. This makes it difficult to plot error bars over an individual β across seeds. To fix this, we use the fact that each experiment has the same number of geometrically spaced β's that start from the same β low and compute statistics averaged across the β's that share an index. The x-axis then represents the β value for each index averaged over seeds.
this section cite: []

Section: D.3.3 Determining the uniqueness of minima
To sample the LSE minima, 500 initial points are uniformly sampled from the support boundary around each memory and gradient descent eq. ( 6) is performed for 13000 steps at a cosine-decayed learning rate α from 0.01 → 0.0001. However, even after this descent process, there are variations in the retrieved memories due to discrete step size α and floating point precision requiring us to be careful when deciding if two samples are distinct. Generally, memory retrieval is said to converge when ∥∇ x E(x)∥ < ϵ for some small ϵ > 0, or when the number of iterations T exceeds some threshold at small α. Because we know the β of the LSE energy, by the properties of the Gaussian kernel we know that two basins merge when their means are within two standard deviations of each
Average Log-Likelihood # of unique generations # of original memories recovered Average Log-Likelihood # of unique generations # of original memories recovered other. Thus, we can say that two distinct samples are generated by the same memory if they are within 2 √ β of each other. When counting the uniqueness of the samples from the LSR energy, we perform the following trick to exactly compute the fixed points of the dynamics. We first compute our "best guess" for the fixed point by performing standard gradient descent according to eq. ( 6) for T steps, at which point z := x (T ) is close (but not exactly equal) to the fixed point x ⋆ . We then pass z to algorithm 1 to compute the fixed point exactly. With a good initial guess z, algorithm 1 converges after a single iteration.
this section cite: []

Section: Algorithm 1: Fixed Point Computation for the LSR Memory Retrieval
Input: Initial guess z, stored patterns {ξ µ } M µ=1 , inverse temperature β Output: Fixed point z ⋆ Initialize previous point z prev ← z + ∞ while z prev ̸ = z do z prev ← z Compute supports near z S(z) ← {ξ µ : ∥z -ξ µ ∥ ≤ 2 β } Update to mean of support centroids z ← 1 |S(z)| ξµ∈S(z) ξ µ end return z
Finally, we choose to sample points near the support boundary of each stored pattern because this maximizes the probability that we will end up in a spurious minimum. The size of spurious basins in high dimension can be very small, and the probability of landing in them decreases rapidly with increasing β (see the region of support plot in fig. 3).
this section cite: []

Section: D.4 Details: Qualitative reconstructions
The experiments in fig. 4 are conducted in two steps. First, we design an energy for each dataset. Then, we discover all memories for each system.
this section cite: []

Section: Designing the energy
The DenseAM energies studied in this work are described by a matrix of stored patterns Ξ and an inverse temperature hparam β.
Choosing Ξ. For MNIST, Ξ is obtained as follows: 24 random images from the MNIST training set are normalized to be [0, 1], rasterized into a 784-dim vector, and projected into a 10-dim latent space of a β-VAE trained according to the methods laid out below. These latents become our stored patterns Ξ MNIST ∈ [0, 1] 24×768 that we use to parameterize both the LSR energy and the LSE energy. The procedure for TinyImagenet [27] is similar. We randomly select 40 images from the dataset, each of shape (C,H,W)=(3,64,64). These samples are passed through a small pretrained VAE called TAESD [28] to produce latents that are of shape (4,8,8) which are then rasterized into vectors of shape (256,). Thus, our stored pattern matrix for TinyImagenet is Ξ TinyImgnet ∈ R 40×256 .
Choosing β Tuning β for the LSR energy is challenging. When β is too small, each stored pattern interacts with all other stored patterns to induce one minimum at the centroid and several minima far from the data distribution; when β is too large, each stored pattern will interact with no other stored patterns and will induce only a single minimum at itself. Neither of these regimes are interesting. For large ranges of β between these limits, the combinatorial search space of possible memories is computationally prohibitive. For this reason, we use a binary search algorithm to choose a β that, on average, causes each stored pattern to interact with approximately 4 other stored patterns (in the 10-dim MNIST case) and 2 other stored patterns in the 256-dim TinyImagenet case. This β encourages the LSR energy to exhibit emergence where it is computationally feasible to enumerate all memories, and we use it for both the LSR and LSE energy experiments. In pseudocode, the search algorithm is given by algorithm 2
Training the β-VAE for MNIST. MNIST images are encoded by a β-VAE [31,32] with a latent dimension of 10. The VAE takes as input the 784-dim rasterized MNIST images. The VAE's encoder and decoder are two layer MLPs configured with LeakyReLU and BatchNorm activations, with a hidden dimension size of 512. Training proceeded for 50 epochs using a learning rate of 1e-3, the Adam optimizer, and a minibatch size of 128. The β of the β-VAE (distinct from the inverse temperature β used by the LSR energy) is set to 4.
this section cite: ['b26', 'b27', 'b30', 'b31']

Section: Discovering all memories
Once we have chosen a β that results in a feasible number of basin interactions K (where the combinatorial search space is tractable), all memories for LSR are discovered by filtering the set of "memory candidates" -the set of centroids formed by at overlapping basins around each stored patterns -to those whose energy gradient is zero. This method is explicitly described in algorithm 3, which iterates through each stored patterns ξ µ and only searches for emergent memories formed by near-enough stored patterns.
this section cite: []

Section: D.6 Additional experiment: Scaling number of stored patterns
The LSR energy function is simple and its properties hold across any scale of stored patterns. To show this, we store all 60,000 MNIST training images into the LSR energy and select a β for which ∼50% of the stored patterns are still retrievable. We select a "seed" image at random and randomly select 15 images from all images whose basins interact with the seed image at the chosen β. This example is shown in Figure 10.
this section cite: []

Section: E Proofs E.1 Proof of Theorem 1
For any x ∈ X , let B(x) = {µ ∈ M : ∥x -ξ µ ∥ 2 ≤ 2/β}. Then the gradient of the LSR energy in eq. ( 3) is given by
∇ x E LSR (x) = M µ=1 (x -ξ µ ) 1 ∥x -ξ µ ∥ 2 ≤ 2 β ϵ + M ν=1 ReLU 1 -β 2 ∥x -ξ ν ∥ 2 (7) = µ∈B(x) (x -ξ µ ) ϵ + ν∈B(x) ReLU 1 -β 2 ∥x -ξ ν ∥ 2(8)
With β = 2/(r -∆) 2 , B(x) = {µ ∈ M : ∥x -ξ µ ∥ ≤ (r -∆)}. For any x ∈ S µ (∆), B(x) = {µ}. Thus the LSR energy gradient simplifies to
∀x ∈ S µ (∆), ∇ x E LSR (x) = (x -ξ µ ) ϵ + ReLU 1 -β 2 ∥x -ξ µ ∥ 2 , (9
)
which is exactly zero at x = ξ µ , thus giving us the retrieval of the µ th memory via energy gradient descent.
Furthermore, again for x ∈ S µ (∆) with a energy gradient descent learning rate set to η ← ϵ + ReLU 1 -β /2 ∥x -ξ µ ∥ 2 , the update is exactly η∇ x E LSR (x) = (x -ξ µ ). Thus a single step gradient descent update to x with x -η∇ x E LSR (x) = x -(x -ξ µ ) = ξ µ results in the retrieval of the µ th memory.
this section cite: []

Section: E.2 Proof of Proposition 2
Proof. (Proposition 2.)
For any x ∈ X , given the definition of B(x), recall that the gradient of the LSR energy is given in eq. ( 7). This gradient is zero when
x = 1 |B(x)| µ∈B(x)
ξ µ , Figure 9: Emergent memories are centroids of subsets of stored patterns, shown clearly when 8 stored images are visualized in pixel space alongside their induced emergent memories. Stored patterns (bottom, indexed A-H) merge to form emergent memories (top, labeled by the stored patterns that merged to form the emergent memory). β is chosen such that the number of emergent memories is approximately the same as the number of stored patterns.
where it is determined by the set B(x) = µ : ∥x -ξ µ ∥ < 2 β .
Let S x * (∆) = {x : ∥x -x * ∥ ≤ ∆} be a basin around an emergent memory x * . Note that by Cauchy-Schwarz inequality, the change within the basin of the squared distance to any ξ µ is at most
|∥x -ξ µ ∥ 2 -∥x * -ξ µ ∥ 2 | ≤ 2∥x * -ξ µ ∥∆ + ∆ 2 .
We now for activated memories µ ∈ B(x * ) let
δ min (x * ) = min µ∈B(x * ) 2 β -∥x * -ξ µ ∥ 2 > 0.
This is the margin to the boundary of activation. Moreover, for inactivated memories ν / ∈ B(x * ), let
γ min (x * ) = min ν / ∈B(x * ) ∥x * -ξ ν ∥ 2 - 2 β > 0.
This is the margin to being activated. Given {ξ µ } M µ=1 has a density, the probability of either of them being exact zero is zero.
Finally, we determine the radius ∆. To ensure that for any x ∈ S x * (∆), ∇ x E LSR (x) = ∇ x E LSR (x * ), it suffices to make sure B(x) = B(x * ) for any x ∈ S x * (∆). To this end, we can pick ∆ such that
2D max (x * )∆ + ∆ 2 < min{δ min (x * ), γ min (x * )}, where D max (x * ) = max µ ∥x * -ξ µ ∥.
This gives
∆ < D 2 max (x * ) + min{δ min (x * ), γ min (x * )} -D max (x * ) =: r * .
To prove the final claim, without loss of generality, we consider the case when X = [0, 1] d with V = 1. Since {ξ µ } M µ=1 are i.i.d. uniform, 1 ξ µ ∈ ∥x -ξ µ ∥ ≤ 2 β are i.i.d. indicators over µ = 1, . . . , M , by the concentration of sub-Gaussian random variables [33, Proposition 2.5], we have: with probability at least
1 -M -1 , λ = |B(x)| ∈   M V d 2 β d 2 -5 M V d 2 β d 2 log M , M V d 2 β d 2 + 5 M V d 2 β d 2 log M   . (11
) This implies λ = Θ M V d 2 β d 2
. Note that according to Proposition 2, for each |B(x)| ∈ (1, M ], there is a corresponding new emergent memory as the average of all stored patterns over µ ∈ B(x). Then, the number of new emergent memories is bounded as
Q β,LSR ≤ M λ , where λ = Θ M V d 2 β d 2 .
By Stirling's formula, we have:
Q β,LSR ≤ eM λ λ .
Thus, it holds that
log Q β,LSR ≤ λ log eM λ = O M V d 2 β d 2 log e V d β 2 d 2 .
This implies
Q β,LSR = O exp M V d 2 β d 2 log e V d β 2 d 2 .
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: The Reproducibility section of the Appendix provides information about the hardware and a description of the code used to create the results reported in the main paper. For important algorithms, we include succinct and clear pseudocode in the Appendix. For review, the anonymized code is submitted alongside the full supplementary as a zip file, while the code is made publicly available on GitHub for the final version.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: We link to a repository that contains code to completely reproduce the results reported in the main paper.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://nips.  cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: The attached code specifies all the training and test details necessary to reproduce the results. Important hyperparameter configurations for the experiment are provided in the paper and appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: Experiments where the results are strongly affected by random seeds are repeated over multiple seeds. The results are averaged and reported alongside errorbars.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: The Reproducibility section of the Appendix provides information about the hardware and resources.
Guidelines:
• The answer NA means that the paper does not include experiments.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: All code and data is properly credited and the licenses properly respected.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: No new assets are introduced in the paper.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve research with human subjects. Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
this section cite: []

Section: 
x 0.0 0.5 1.0 K(x) Gaussian exp( x 2 ) Efficiency: 95.1% 3 2 1 0 1 2 3 x Triangle ReLU(1 x ) Efficiency: 98.6% 3 2 1 0 1 2 3 x Uniform I(1 x 2 ) Efficiency: 92.9% 3 2 1 0 1 2 3 x Epanechnikov ReLU(1 x 2 ) Efficiency: 100.0% 3 2 1 0 1 2 3 x Cosine cos( 2 min( x , 1)) Efficiency: 99.9% 3 2 1 0 1 2 3 x Quartic ReLU(1 x 2 ) 2 Efficiency: 99.4% 3 2 1 0 1 2 3 x Triweight ReLU(1 x 2 ) 3 Efficiency: 98.7% 3 2 1 0 1 2 3 x
Tricube ReLU(1 x 3 ) 3  Efficiency: 99.8%
Figure 5: Different kernels used in KDE with their expression and KDE efficiency relative to the Epanechnikov kernel (higher is better, see text for details). The center of each kernel is marked with a red ⋆. To highlight the shape of the kernel, we have removed any scaling in the kernel expression. Note that all above kernels except Gaussian have finite support. The Epanechnikov kernel has the highest efficiency (100%). While the Gaussian kernel is extremely popular, and it is more efficient (95.1%) than the Uniform kernel (92.9%), there are various other kernels with better efficiency.
1. Triangle kernel -The triangle kernel exhibits very interesting emergence behavior. A perfectly flat energy manifold is formed where two or more basins overlap. If the energy of this manifold is lower than the energy of the stored patterns, the stored patterns "merge" and are no longer retrievable. If the energy of the manifold is higher, both original patterns are preserved and we observe a local emergent manifold of memory. 2. Uniform kernel -A uniform kernel produces an energy landscape that is impractical for associative memory because the gradient is zero everywhere except at the discontinuities. Emergent minima exist according to Definition 3 and have exclusively lower energy than the stored patterns. 3. Triweight kernel -We were not able to find emergence at any temperature with the triweight kernel, despite its compact support. Indeed, looking at the β = 0.19 row of Figure 6, we see that the transition of two basins merging results in a single, almost flat minimum. This phenomenon of compact support without emergence (or perhaps, emergence that is limited to an extremely narrow range of β that we were not able to find) requires further investigation. 4. Quartic kernel -The quartic kernel produces smoother energy landscapes than the Epanechnikov does, and emergence appears within a very narrow range of β. Unlike the Epanechnikov kernel, overlapping basins do not guarantee emergent minima, and the emergent minimum is unlikely to have lower energy than the stored patterns. 5. Tricube kernel -The tricube kernel behaves like the Quartic kernel, but emergent memories are likely to have lower energy than the stored patterns. The Tricube kernel has an interesting property where local minima flatten right before basins merge. The resulting energy landscape is smoother than that of the Epanechnikov kernel. 6. Cosine kernel -The cosine kernel looks remarkably like the Epanechnikov kernel, and its emergence properties are almost identical to that of LSR. However, it appears that Cosine-emergent memories have slightly higher energy than their Epanechnikov counterpart.
this section cite: []

Section: D Experimental Details

this section cite: []

Section: D.1 Reproducibility & Technical Resources
The codebase is published in a GitHub repository and contains necessary instructions to setup the environment and to recreate all results, down to the same seed used for training and sampling. Experiments use both PyTorch [29] and JAX [30]. Each experiment was run on a system with access to 8xA40 GPUs each with 40GB of memory. The log-likelihood experimental sweep in appendix D.3 took ∼12 hours using an optimized scheduler across 45 available CPU cores (memory turned out to be the bottle neck and the CPUs had access to much more RAM than the GPU). Searching for all novel minima at different β and M in appendix D.2 took ∼48 hours using an optimized scheduler across all 8 GPUs. Training the VAE in appendix D.4 took < 30 min on a single GPU; enumerating all local minima using the efficient algorithm 2 and algorithm 3 took < 15 min.
this section cite: ['b28', 'b29']

Section: D.2 Details: Quantifying Novel Minima
In this experiment we tested across a geometrically spaced range of β ∈ [2d -1 , 2r -2 min ], where r min := min µ̸ =ν ∥ξ µ -ξ ν ∥ is the minimum pairwise distance between any two stored patterns in the current subset K ⊆ M . At the largest β, the support regions of the stored patterns are disjoint and the only memories are the M stored patterns themselves; this configuration has a very small support region (shown as the shaded green curve in fig. 3, which is computed by monte carlo sampling 1e6 Algorithm 2: Binary Search over β to achieve desired memory interactions Input: Desired avg. interactions per pattern K, stored patterns {ξ µ } M µ=1 , max iterations n max Output: Optimal β * achieving number of basin interactions K Compute Distance matrix D µν ← {∥ξ µ -ξ ν ∥} Binary bounds (r min , r max ) ← (0.5 min(D), 4 max(D)) Initial basin radius r ← mean(r min , r max ) end n iter ← 0 repeat Compute avg. number of interacting basins per memory
K ′ ← mean µ M ν=1 1[D µν ≤ 2r] Update binary search conditions r min ← r if K ′ < K r max ← r if K ′ > K r ← mean(r min , r max ) n iter ← n iter + 1 until K ′ = K or n iter ≥ n max β * ← 2/r 2 return β *
Algorithm 3: Discover local minima of the LSR energy at a specific β.
Input: Stored patterns {ξ µ } M µ=1 , inverse temperature β, gradient norm threshold δ near 0 Output: Set of LSR memories X * . Compute Distance matrix D µν ← {∥ξ µ -ξ ν ∥} Basin radius r ← 2/β end Initialize set of local minima X * ← ∅ for µ ∈ M do Compute set of interacting neighbors X µ ← {ξ ν : D µν ≤ 2r} Compute the set of all non-empty subsets C ← {S ⊆ X µ : size(S) > 0} for S ∈ C do Compute centroid of neighbors xS ← mean(S) Compute xS neighbors T S ← {ξ ν : ∥ξ ν -xS ∥ ≤ r} if ∥∇E LSR (x S )∥ < δ & E LSR (x S ) < ∞ then Update set of local minima X * ← X * ∪ {x S } end end end return X *
All LSE memories are discovered via gradient descent. We initialize queries x µ = ξ µ and perform gradient descent according to eq. ( 6) until convergence (for this experiment, we iterated for 20k steps at small step-size α = 0.002). The retrieved fixed points are the complete set of LSE memories.
this section cite: []

Section: D.5 Additional experiment: Pixel-space emergence
When the stored patterns live in a semantically structured latent space, as is done in fig. 4, the energy landscape inherits the structure such that centroids of subsets of stored patterns appear semantically novel. However, the latent space visually obscures the mechanistic simplicity of emergent memories. Thus, we repeat the experiment in pixel space to reinforce how emergent memories work.
We store 8 randomly selected MNIST images as rasterized pixels (normalized between 0 and 1) into the LSR energy. The resulting emergent minima and stored patterns are shown in Figure 9, where the β is chosen to balance the number of emergent minima and the retrievability of the stored patterns (global emergence Definition 2).
this section cite: []

Section: Seed image
Nearby emergent memories the geometric mean of the memories corresponding to the set B(x). Moreover, by standard algebraic computations, we have
∇ 2 x E LSR (x * ) = |B(x * )| ϵ + M ν=1 ReLU 1 -β 2 ∥x * -ξ ν ∥ 2 I d ≻ 0.
this section cite: []

Section: E.3 Proof of Theorem 2
Note that once we prove properties (a) and (b), the ε-global emergence in Definition 2 follows immediately.
We first focus on (a). Note that for any fixed x ∈ X and
ε > 0, vol(B(x, ε) ∩ X ) is at most vol(B(x, ε)) = V d ε d
, where V d is the volume of the unit ball in R d . Thus, we obtain: for any pair (ξ µ , ξ ν ) with µ ̸ = ν, we have:
Pr ∥ξ µ -ξ ν ∥ < ε ≤ X vol(B(x, ε) ∩ X ) vol(X ) dx ≤ V d V -1 ε d .
Then, we have:
Pr min 1≤µ̸ =ν≤M ∥ξ µ -ξ ν ∥ < ε ≤ M 2 V d V -1 ε d .(10)
Now, we let ε = (V d /V ) -1/d e -2α for a positive α. Thus, with probability at least 1 -M 2 e -2αd , the minimum pairwise Euclidean distance r = min 1≤i̸ =j≤M ∥ξ
µ -ξ ν ∥ ≥ (V d /V ) -1/d e -2α
. Then, with
0 < β = 2 (r -∆) 2 ≤ 2 ((V d /V ) -1/d e -2α -∆) 2 ,
we are able to retrieve any memory ξ µ with an x ∈ S µ (∆), for ∆ ∈ (0, (V d /V ) -1/d e -2α ). Moreover, the size of M is given by the success probability:
δ = 1 -M 2 e -2αd .
this section cite: []

Section: This gives
M = Θ √ 1 -δe αd .
Next, we focus on (b). Recall the gradient is given by
∇ x E LSR (x) = µ∈B(x) (x -ξ µ ) ϵ + µ∈B(x) ReLU 1 -β 2 ∥x -ξ µ ∥ 2 ,
this section cite: []

Section: E.4 Proof of Proposition 3
Without loss of generality, we consider the case when X = [0, 1] d with V = 1. Now, we divide [0, 1] d into an equally spaced grid of size M , i.e., we divide each dimension [0, 1] into [mM -1/d , (m + 1)M -1/d ] for m = 0, . . . , M -1. Under our grid setting, {ξ µ } M µ=1 lie in such a grid of equal size. Now, note that
B(x) = µ : ∥x -ξ µ ∥ < 2 β .
Suppose the ball in definition of B(x) contains λ of those equally spaced stored patterns such that λ = |B(x)| = λ ∈ (1, M ]. We want to put a hypercube in the grid such that it contains exact λ out of M equally spaced stored patterns in the grid. Here, for simplicity, it is up to a constant depending on d if we think of the ball as a hypercube in R d containing the points in the grid in this case. Hence, in each dimension 1 ≤ i ≤ d, we put an interval containing of order λ 1/d points in the grid inside the entire interval containing of order M 1/d points in the grid. It thus gives the number of choice for each dimension 1 ≤ i ≤ d as:
M 1/d -λ 1/d -1
Hence, counting all choices over d dimensions, we obtain the number of such B(x) as
M 1/d -λ 1/d -1 d .
Moreover, due to the equal spacing of the grid, we have:
λ = Θ M 8 β d 2 .(12)
Hence, combining all bounds above, we obtain: the number of emergent memories under uniform sampling is of order
Θ M 1/d -λ 1/d + 1 d ,
where λ satisfies equation 12 and 1 < λ ≤ M .
this section cite: []

Section: E.5 Proof of Proposition 1
Recall the LSE energy:
E LSE (x) = - 1 β log M µ=1 e -β 2 ∥x-ξµ∥ 2 .
Thus, the gradient is
∇ x E LSE (x) = x - M µ=1 p µ (x)ξ µ , p µ (x) := e -β 2 ∥x-ξµ∥ M µ=1 e -β 2 ∥x-ξµ∥ .
Suppose ξ ν for some ν is the zero of this gradient, it yields that
(1 -p ν (ξ ν ))ξ ν = µ̸ =ν p µ (ξ µ )ξ µ .
This is a contradiction of the assumption that {ξ µ } M µ=1 are linearly independent.
this section cite: []

Section: NeurIPS Paper Checklist
1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The abstract and introduction make claims that are clearly backed by theorems and experiments in the paper. Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: Limitations are discussed in the limitations section of the supplementary.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model wellspecification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: Each theorem provides the full set of assumptions and a complete (and correct) proof in the Proofs section of the Appendix. Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: In publishing this paper we have adhered in full to the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: This work is primarily theoretical and considers only toy examples (MNIST, synthetic data, TinyImagenet). It will have no societal impact.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 11.
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: There is no risk of misuse of the contributions of this paper.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The paper does not involve LLMs as any important, original, or non-standard components. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: ['b15']

Section: References
Ref_id:b0 Title: Dense associative memory for pattern recognition Year: (2016)
Ref_id:b1 Title: A new frontier for hopfield networks Year: (2023)
Ref_id:b2 Title: Dense associative memory through the lens of random features Year: (2024)
Ref_id:b3 Title: Neural networks and physical systems with emergent collective computational abilities Year: (1982)
Ref_id:b4 Title: Dense associative memory is robust to adversarial inputs Year: (2018)
Ref_id:b5 Title: On a model of associative memory with huge storage capacity Year: (2017)
Ref_id:b6 Title: Hopfield networks is all you need Year: (2021)
Ref_id:b7 Title: Large associative memory problem in neurobiology and machine learning Year: (2021)
Ref_id:b8 Title: Exponential capacity of dense associative memories Year: (2024)
Ref_id:b9 Title: Hierarchical associative memory Year: (2021)
Ref_id:b10 Title: A universal abstraction for hierarchical hopfield networks Year: (2022)
Ref_id:b11 Title: On sparse modern hopfield model Year: (2023)
Ref_id:b12 Title: Sparse and structured hopfield networks Year: (2024)
Ref_id:b13 Title: Uniform memory retrieval with larger capacity for modern hopfield models Year: (2024-07)
Ref_id:b14 Title: Does data interpolation contradict statistical optimality? Year: (2019)
Ref_id:b15 Title: Fit without fear: remarkable mathematical phenomena of deep learning through the prism of interpolation Year: (2021)
Ref_id:b16 Title: Deep learning: a statistical viewpoint Year: (2021)
Ref_id:b17 Title: Kernel smoothing Year: (1994)
Ref_id:b18 Title: Combinatorial methods in density estimation Year: (2001)
Ref_id:b19 Title: Non-parametric estimation of a multivariate probability density Year: (1969)
Ref_id:b20 Title: Smooth optimum kernel estimators of densities, regression curves and modes. The Annals of Statistics Year: (1984)
Ref_id:b21 Title: On qualitative smoothness of kernel density estimates Year: (1995)
Ref_id:b22 Title: On the number of modes of gaussian kernel density estimators Year: (2024)
Ref_id:b23 Title: Statistical neurodynamics of associative memory Year: (1988)
Ref_id:b24 Title: A robust method for distinguishing between learned and spurious attractors Year: (2004)
Ref_id:b25 Title: Phase diagram of restricted boltzmann machines and generalized hopfield networks with arbitrary priors Year: (2018)
Ref_id:b26 Title: Tiny imagenet visual recognition challenge Year: (2015)
Ref_id:b27 Title: Tiny autoencoder for stable diffusion Year: (2023)
Ref_id:b28 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b29 Title: JAX: composable transformations of Python+NumPy programs Year: (2018)
Ref_id:b30 Title: Auto-encoding variational bayes Year: (2022)
Ref_id:b31 Title: beta-VAE: Learning basic visual concepts with a constrained variational framework Year: (2017)
Ref_id:b32 Title: High-dimensional statistics: A non-asymptotic viewpoint Year: (2019)
