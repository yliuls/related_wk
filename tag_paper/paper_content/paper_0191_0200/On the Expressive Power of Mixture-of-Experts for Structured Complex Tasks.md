Title: On the Expressive Power of Mixture-of-Experts for Structured Complex Tasks
Abstract: Mixture-of-experts networks (MoEs) have demonstrated remarkable efficiency in modern deep learning. Despite their empirical success, the theoretical foundations underlying their ability to model complex tasks remain poorly understood. In this work, we conduct a systematic study of the expressive power of MoEs in modeling complex tasks with two common structural priors: low-dimensionality and sparsity. For shallow MoEs, we prove that they can efficiently approximate functions supported on low-dimensional manifolds, overcoming the curse of dimensionality. For deep MoEs, we show that O(L)-layer MoEs with E experts per layer can approximate piecewise functions comprising E L pieces with compositional sparsity, i.e., they can exhibit an exponential number of structured tasks. Our analysis reveals the roles of critical architectural components and hyperparameters in MoEs, including the gating mechanism, expert networks, the number of experts, and the number of layers, and offers natural suggestions for MoE variants.

Section: Introduction
Mixture-of-experts (MoE) models (Jacobs et al., 1991;Jordan and Jacobs, 1994) have recently achieved significant success in deep learning, particularly as a core architectural component of modern large language models (LLMs) (Abdin et al., 2024;Yang et al., 2024b;Liu et al., 2024;Cai et al., 2025). These models have demonstrated strong capabilities across a wide range of complex and diverse tasks, including mathematical reasoning, logical inference, language understanding, and code generation. Despite their empirical success, the theoretical foundations underlying MoEs remain poorly understood, especially in their capacity to efficiently model complex tasks.
In both machine learning and applied mathematics, it is widely recognized that although real-world tasks may appear complex, they often exhibit latent structures. Two prominent structural priors are: (1) low-dimensional structure: high-dimensional data typically lies on a manifold of much lower intrinsic dimension; (2) sparse structure: meaningful signals tend to admit sparse representations in suitable bases or dictionaries. These structural priors have motivated numerous influential algorithms, including dimensionality reduction (Tenenbaum et al., 2000), sparse regression via Lasso (Tibshirani, 1996), compressed sensing (Donoho, 2006), and neural network pruning and compression techniques.
In this work, we investigate the expressive power of MoE networks for modeling complex tasks that exhibit either low-dimensional or sparse structure. Our contributions are summarized as follows:
• Shallow MoE networks. We prove that shallow MoE networks can efficiently approximate functions supported on low-dimensional manifold. Theoretically, this task reduces to a collection of simpler approximation subproblems localized on low-dimensional subregions, along with an assignment problem that maps each input to the appropriate region. We show that shallow MoE networks naturally implement this procedure, thereby avoiding the curse of dimensionality.
Our analysis reveal the complementary roles of the two core components in MoE: expert networks approximate localized subfunctions, while the gating mechanism ensures correct input-to-expert assignment. Additionally, the analysis offers practical suggestions on MoE variants, such as the nonlinear gating, alternating MoE architectures with equivalent expressivity, and low-dimensional expert networks with auto-encoding.
• Deep MoE networks. We formalize complex tasks as piecewise functions, and focus on a broad class of structured tasks exhibiting compositional sparsity: where each subtask depends on only a small subset of input coordinates, and the overall task is a hierarchial composition of these subtasks. We demonstrates that a depth-O(L) MoE network with E expert per layers can efficiently approximate piecewise functions with E L distinct pieces, i.e, it can exhibit an exponential number of structured tasks. Moreover, our analysis elucidates the distinct roles of network depth L (which enables hierarchical composition) and expert count E (which enables subtask specialization).
• Unified insights. Our theoretical results reveal that MoE networks can effectively discover the underlying structure priors in the complex tasks (such as low-dimensionality or sparsity), and subsequently decompose them into simpler subproblems, each solved by specialized experts.
2 Related Works Theoretical understanding of MoE. Chen et al. (2022) analyzed the training dynamics of shallow MoE networks with softmax gating on clustered datasets, emphasizing the importance of expert nonlinearity and data structure. (Baykal et al., 2022) showed that sparsely activated networks can achieve approximation performance comparable to dense networks, and offered a computationally efficient alternative. Dikkala et al. (2023) examined the impact of learnable routing mechanisms in MoEs, establishing their benefits. Li et al. (2024) investigated MoE in continual learning, using overparameterized linear regression to show their adaptability across tasks. A comprehensive survey of recent theoretical advances is presented in Mu and Lin (2025). In contrast to these prior works, we focus on the expressive power of both shallow and deep MoE networks for broad classes of structured functions.
Low-dimensional structure. The manifold hypothesis posits that high-dimensional data in real world (e.g., images, speech, and text) typically lies on a manifold of much lower intrinsic dimensionality than the ambient space. This perspective motivates various algorithmic approaches: (i) Dimensionality reduction techniques (Tenenbaum et al., 2000;Roweis and Saul, 2000;Belkin and Niyogi, 2003), which aim to uncover and utilize such low-dimensional structures. (ii) Representation learning methods like Autoencoders and Variational Autoencoders (Hinton and Salakhutdinov, 2006;Kingma et al., 2013), which seek compact and informative representations aligned with low-dimensional manifold.
Sparse structure. It is widely believed that meaningful signals often admit sparse representations in appropriate bases or dictionaries. This principle underpins many influential algorithms, such as Lasso (Tibshirani, 1996), Compressed Sensing (Donoho, 2006;Candès and Wakin, 2008), and Sparse Coding (Olshausen and Field, 1996;Elad and Aharon, 2006), which have been widely applied across domains.
From a theoretical standpoint, the prevalence of sparsity and low-dimensionality has inspired recent studies on the expressive power of deep networks under structural assumptions. For example, Mhaskar and Poggio (2016); Poggio (2023) analyzed dense neural networks approximating functions with compositional sparsity, demonstrating how sparsity mitigates the curse of dimensionality (Bellman, 1966;Bach, 2017). Wang et al. (2024) studied the expressivity of Transformer models (Vaswani et al., 2017) for modeling long but sparse memories, showing the model's capacity to overcome the curse of memory. Shaham et al. (2018); Chen et al. (2019) examined the approximation power of dense networks for functions supported on low-dimensional manifolds. In contrast to these works, we investigates the expressive power of MoE networks in approximating complex functions exhibiting either sparse or low-dimensional structure.
3 Preliminaries Basic notations. Let f : Ω → R be a continuous function defined on a compact set Ω. Its L ∞ norm is defined as ∥f ∥ L∞ := sup x∈Ω |f (x)|. We use standard asymptotic notations O(•), Ω(•), Θ(•) to hide the constants independent of the primary problem size (typically denoted by m), and the notations Õ(•), Ω(•), Θ(•) further hide logarithmic factors. For a positive integer n, let [n] = {1, • • • , n}. For a, b ∈ R, define a∧b = min{a, b} and a∨b = max{a, b}.
this section cite: ['b17', 'b18', 'b0', 'b39', 'b23', 'b6', 'b34', 'b35', 'b11', 'b9', 'b3', 'b10', 'b22', 'b27', 'b34', 'b32', 'b4', 'b16', 'b19', 'b35', 'b11', 'b7', 'b30', 'b13', 'b26', 'b31', 'b5', 'b2', 'b37', 'b36', 'b33', 'b8']

Section: MoE networks
MoE components. An MoE layer consists of two primary components:
• Expert networks: A collection of E expert networks, each implemented as a dense feedforward ReLU neural network:
f (1) , • • • , f (E) : R din → R dout .
• Gating network: A gating function g : R din → R E . In most existing MoE models (Fedus et al., 2022;Du et al., 2022;Yang et al., 2024a), g is linear due to its simplicity and empirical effectiveness: g(x) = W R x, where W R ∈ R E×din .
this section cite: ['b15', 'b12', 'b38']

Section: MoE operation.
Given an input x ∈ R din , an MoE layer performs the following operations, as illustrated in Figure 1:
• Expert selection. The gating network computes routing scores g(x) ∈ R E , and selects the top-K experts with the highest scores:
K := arg TopK(g(x)),
where arg TopK(z) returns the indices of the K largest entries of z.
• Expert computation and aggregation. Each selected expert k ∈ K computes its output f (k) (x). The final output is a weighted combination:
y = k∈K α k (x)f (k) (x),
where the weight are defined via
α k (x) = exp(g k (x)) j∈K exp(gj (x)) .
Notably, only K expert networks are activated per input. Without loss of generality, we focus throughout this paper on the case K = 1, as the extension to arbitrary K ⩽ E is straightforward.
Hypothesis class H L,E l,m . We define H L,E l,m as the class of depth-L neural networks composed of stacked L MoE layers: h (L) • h (L-1) • • • • • h (1) ,(1)
where each h (ℓ) is an MoE layer consisting of a linear gating network g (ℓ) and E expert networks f (ℓ,e) (e ∈ [E]), each being an l-layer, m-width dense ReLU neural network.
this section cite: []

Section: Classical approximation results

this section cite: []

Section: Approximation error notation. Let E FFN
l,m (f ) denote the L ∞ approximation error of a target function f : Ω → R using l-layer, m-width dense ReLU neural networks. C K space. Let D, K ∈ N, and Ω ⊂ R D be compact. The space C K (Ω) consists of all functions f such that
∥f ∥ C K (Ω) = max 0⩽∥β∥1⩽K ∥D β f ∥ L∞(Ω) < ∞(2)
where D β f denotes the partial derivatives of order β = (β 1 , • • • , β D ) ∈ Z D + . The space of smooth functions is defined as C ∞ (Ω) = K∈N C K (Ω). Additionally, the smoothness exponent of f : Ω → R is defined as κ(f ) := sup{K ∈ N : f ∈ C K (Ω)}.
The following result summarizes the classical approximation rate of two-layer ReLU networks for C K functions (Mao and Zhou, 2023;Yang and Zhou, 2024):
Theorem 3.1. Let D, K ∈ N, and Ω ⊂ R D be compact. For any f ∈ C K (Ω) and m ∈ N, there exits a two-layer ReLU neural network f m with m hidden neurons such that
E FFN 2,m (f ) ⩽ ∥f -f m ∥ L∞(Ω) ⩽ O m -K D , if K < D+3 2 , Õ m -1 2 , otherwise.
When the smoothness of the target function is relatively low, i.e., K ≪ D, the approximation rate O m -K D reveals that two-layer networks suffer from the curse of dimensionality (CoD).
this section cite: ['b25', 'b40']

Section: Theory for Shallow MoE Networks
In this section, we study the efficiency of shallow MoE networks in approximating functions supported on a low-dimensional manifold M.
this section cite: []

Section: Manifold in Euclidean Space
Figure
2: A d-dimensional manifold M in R D .
Let M be a d-dimensional smooth manifold embedded in R D . We begin by reviewing several standard definitions.
Definition 4.1 (Chart and Atlas).
• A chart for M is a pair (U, ϕ) such that U ⊂ M is open and ϕ : U → R d , where ϕ is a homeomorphism (i.e., bijective, ϕ and ϕ -1 are both continuous). U is called a coordinate neighborhood, and ϕ is the associated coordinate map.
• An atlas of M is a collection {(U α , ϕ α )} α∈A of charts such that ∪ α∈A U α = M.
An atlas {(U α , ϕ α )} α∈A is called smooth if for any overlapping charts (U α , ϕ α ) and (U α ′ , ϕ α ′ ), the transition maps ϕ α • ϕ -1 α ′ and ϕ α ′ • ϕ -1 α are smooth functions. Definition 4.2 (Smooth manifold). The manifold M is called smooth if it has a smooth altas.
We now introduce the partition of unity, which can divide the manifold into regular subregions. Definition 4.3 (Partition of unity). Let {U α } α∈A be an open cover of M. A partition of unity of M w.r.t this cover is a family of nonnegative smooth functions ρ α : M → [0, 1] for α ∈ A such that:
• (i) for all α ∈ A, ρ α has compact support and supp(ρ α ) ⊂ U α ;
• (ii) for every x ∈ M, only finitely many ρ α (x) are nonzero;
• (iii) for all x ∈ M, α∈A ρ α (x) = 1.
Theorem 4.4 (Existence of a partition of unity). Let {U α } α∈A be an open cover of a smooth manifold M. Then there exists a partition of unity {ρ α } α∈A of M w.r.t. {U α } α∈A .
We next define the smoothness of a function defined on a manifold. Definition 4.5 (Function on the manifold). Let a function f : M → R, and {(U α , ϕ α )} α∈A be a smooth atlas of M. Its smoothness κ(f ) is defined by κ(f ) := inf α∈A κ(f • ϕ -1 α ), where the smoothness of each f • ϕ -1 α is defined as Equation (3).
In this paper, we focus on smooth compact manifolds. Due to the compactness of M, its atlas consists of a finite collection of charts, denoted by {(U i , ϕ i )} i∈ [E] . Additionally, we can let ϕ i (U i ) ⊂ [0, 1] d . By Theorem 4.4, there exists a corresponding partition of unity {ρ i } i∈ [E] .
Compact smooth manifolds admit an atlas with strong geometric regularity as below, which is detailed in Appendix A.
Example 4.6 (Highly regular atlas). Let M be a compact smooth manifold. Then there exists a highly smooth atlas {(U i , ϕ i )} i∈[E] such that each map ϕ i : U i → [0, 1] d is a linear function. Thus, each ϕ i satisfies κ(ϕ i ) = ∞ (when viewed as a function in R D ).
Motivated by this example, we define a broad class of regular atlas:
Definition 4.7 (Regular atlas). A atlas {(U i , ϕ i )} i∈[E] of compact manifold M is called regular, if each map ϕ i has the smoothness κ(ϕ i ) > D+3 2 .
this section cite: []

Section: Theoretical results and insights
Theorem 4.8 (Main result). Let M be a compact, d-dimensional smooth manifold in R D , with a regular atlas {(U i , ϕ i )} i∈[E] (Definition 4.7). Let the target function f : M → R. Then for any m ⩾ Ω E 2 , there exists a depth-2 MoE network Ψ ∈ H 2,E 3,m , with E experts per layer, each being is a 3-layer m-width dense networks, such that:
∥f -Ψ∥ L∞(M) ⩽ max i∈[E] E FFN 3,m f | Ui ⩽ max i∈[E] E FFN 2,m f | Ui • ϕ -1 i approximate a low-dimensional function + max i∈[E] E FFN 2,m ϕ i approximate a high-order smooth map f | Ui • ϕ -1 i C 1 ([0,1] d ) ⩽ max i∈[E] Õ m - κ(f | U i ) d ∧ 1 2 .
Theorem 4.8 demonstrates that depth-2 MoE networks can efficiently approximate functions supported on low-dimensional manifolds. The total approximation error decomposes into two components: (i) approximation of low-dimensional target functions f | Ui • ϕ -1 i 's; (ii) approximation of smooth coordinate maps ϕ i 's. Both subproblems are significantly simpler than approximating the original high-dimensional function directly, enabling MoE networks to overcome the curse of dimensionality. Table 1: Comparison of approximation rates between shallow MoE and shallow dense networks. For MoE, m is the width of each expert networks; for dense networks, m is the width of the hidden layer. Shallow MoE networks (Theorem 4.8) Shallow dense networks (Theorem 3.1) max i∈[E]
Õ m - κ(f | U i ) d ∧ 1 2 Õ m -κ(f ) D ∧ 1 2
Improved efficiency over dense networks.
Table 1 highlights the superior approximation efficiency of MoEs. In the regime where the target f has limited smoothness, i.e., κ(f ) ≪ D, dense networks suffer from the curse of dimensionality, as the approximation rate κ(f )/D deteriorates with ambient dimension D. In contrast, MoE networks achieve rates governed by the intrinsic dimension d ≪ D and local smoothness κ(f | Ui ) ⩾ κ(f ), thereby substantially improving approximation efficiency and achieving faster approximation rate:
κ(f | U i •ϕ -1 i ) d ∧ 1 2 ≫ κ(f ) D .
this section cite: []

Section: Key insight. The proof of Theorem 4.8 (deferred to Appendix A) reveals several insights into the mechanisms of MoE networks:
MoE networks achieve efficient approximation by decomposing a complex approximation problem into multiple localized approximation subproblems, as well as a simple assignment task.
Recall that a depth-2 MoE (Eq. ( 1)) comprises expert networks and a gating mechanism. Their distinct roles are as follows:
• Expert networks (f (2,i) in Layer 2): these components efficiently approximate the d-
dimensional local target function f | Ui • ϕ -1 i
and the smooth chart map ϕ i . They directly influence the overall approximation error and benefit from increased width m.
• Routing mechanism (Layer 1 and gating in Layer 2): these components work together to exactly assign each input to its correct expert f (2,i) . The first MoE layer h (1) behaves like a dense model, approximating the smooth partition functions ρ i 's. Then the Layer-2 gating network g (2) selects the expert corresponding to the region U i such that x ∈ U i . This exact assignment is nontrivival, please refer to the proof for details. Corollary 4.9 (Special case, highly regular atlas). Let M be a compact, d-dimensional smooth manifold in R D , with a highly regular atlas {(U i , ϕ i )} i∈[E] (Example 4.6). Let the target function f : M → R. Then for any m ⩾ Ω E 2 , there exists a depth-2 MoE network Ψ ∈ H 2,E 2,m , with E experts per layer, each a 2-layer m-width dense network, such that:
∥f -Ψ∥ L∞(M) ⩽ max i∈[E] E FFN 2,m f | Ui • ϕ -1 i approximate a low-dimensional function ⩽ max i∈[E] Õ m - κ(f | U i ) d ∧ 1 2 .
Compared to Theorem 4.8, Corollary 4.9 applies to a more structured setting in which the local coordinate maps ϕ i are linear and thus thus do not require approximation. This eliminates the second term in the error bound, and reduces each expert to a 2-layer dense network.
Comparison with prior works Shaham et al. (2018); Chen et al. (2019). These works show that dense networks can also efficiently approximate functions on low-dimensional manifolds. However, their analyses require additional regularity assumptions on manifolds, which enables explicit constructions of coordinate charts and partition functions. In contrast, our Theorem 4.8 does not require such explicit formulations, applying to a broader class of smooth regular manifolds. More importantly, MoEs offer a fundamental computational advantage: while dense networks activate all parameters for every input, MoEs selectively activate only a single expert per input. To achieve a comparable approximation accuracy, the number of activated parameters in dense networks is roughly E times greater than that in MoEs, as dense networks must simultaneously approximate all E subproblems.
this section cite: ['b33', 'b8']

Section: Practical Suggestions
Beyond theoretical guarantees, our analysis also offers several practical suggestions into the design of MoE architectures.
Incorporating nonlinearity into gating is critical. Our theoretical results indicate that accurate input-to-expert assignment requires approximating the partition functions ρ i , which are generally nonlinear. Since standard gating function is linear and lacks the capacity to model nonlinear ρ i , an additional MoE (or dense) layer is needed prior to gating to approximate ρ i . If the router incorporates sufficient nonlinearity, e.g., a two-layer ReLU routing network, it can directly model complex partitions, reducing the depth and number of parameters. For instance, in Theorem 4.8, the required depth will reduce from 2 to 1, because the nonlinearity in router eliminates the need for a preceding MoE layer. Similarly, in Theorem 5.2, the required depth will reduce from 2L to L. Therefore, a more direct and potentially more efficient alternative is to incorporate nonlinearity directly into the gating network, eliminating the need for a preceding MoE layer. This observation is consistent with recent empirical findings (Zuo et al., 2021;Liu et al., 2022;Nguyen et al., 2024;Akbarian et al., 2024;Le et al., 2024), which demonstrate that nonlinear gating functions improve MoE performance.
Alternating MoE architectures with equivalent expressive power. In our construction, the first MoE layer actually serves as a standard dense layer with width Ω(E 2 ) to approximate the partition functions ρ i . This insight motivates alternative architectures with comparable expressive power: (i) MoE-dense alternating networks. A natural variant consists of alternating dense and MoE layers, e.g., h moe • h dense . This design extends naturally to deeper architectures. This design has been adopted in practice by GShard (Lepikhin et al., 2020) and GLAM (Du et al., 2022). (ii) Shared + routed experts per MoE layer. Another common MoE variant incorporates one shared expert alongside E routed experts. This structure is empirically adopted in modern MoE architectures such as Qwen2 (Yang et al., 2024b) and DeepSeek (Liu et al., 2024).
Low-dimensional expert networks via autoencoding. Our analysis also suggests a more structured and interpretable design for expert networks in MoE. Typically, each expert is implemented as a dense network with input dimension D and width O(D), resulting in O(D 2 ) parameters. However, our theory motivates replacing each expert f (2l,i) with a composition f low •Enc, where: Enc : R D → R d is an encoder approximating the smooth coordinate chart ϕ :
U i → [0, 1] d , f low : R d → R is a low-dimensional dense network approximating f l,i •ϕ -1 .
This design reduces the number of trainable parameters in each expert to #(Enc) + O(d 2 ), which is significantly smaller than O(D 2 ) when d ≪ D. Moreover, this decomposition aligns with the manifold structure of the target function, improving interpretability. To support encoder learning, one can incorporate a standard reconstruction loss E x ∥Dec(Enc(x)) -x∥ 2 2 using a decoder Dec. We leave empirical validation of this theoretically motivated architecture for future work.
5 Theory for Multi-layer MoE Networks Piecewise functions as multiple tasks. Modern LLMs are capable of performing a wide range of tasks, such as mathematics, logical reasoning, language understanding, and code generation. From a mathematical perspective, each task can be viewed as a function defined on a task-specific input domain. In practice, these tasks often differ, which are supported on distinct regions Ω 1 , • • • , Ω N , with distinct corresponding tasks f i : Ω i → R. Therefore, performing N tasks can be naturally modeled as approximating a piecewise function:
f (x) = f i (x), if x ∈ Ω i , i ∈ [N ].
While each f i may be high-order smooth within its region Ω i , the global function f may exhibit only low-order smoothness at the interfaces between adjacent regions.
Key question. Theorem 4.8 shows that a depth-2 MoE network with E experts per layer can efficiently approximate a piecewise function f comprising E pieces (f | Ui , i ∈ [E]) (each handled by a different expert). A natural question then arises for deep MoE networks: How many distinct pieces can be efficiently modeled by a deep MoE network?
A naive limitation. As illustrated above, it is intuitive to associate each expert with a distinct task. A depth-L MoE with E experts per layer contains O(LE) experts in total, implying a capacity to model at most O(LE) distinct regions if each expert is used independently.
Overview of our result: beyond the naive limitation. Surprisingly, this limitation can be overcome when the target function exhibits compositional sparsity. We will show that: Depth-O(L) MoE networks with E experts per layer can efficiently approximate a piecewise function comprising E L pieces, provided the function satisfies a compositional sparsity structure.
This demonstrates that MoE networks can model an exponential number of structured tasks (far surpassing the native limitation O(LE)) by exploiting structured sparsity in the function.
this section cite: ['b42', 'b28', 'b20', 'b21', 'b12', 'b39', 'b23']

Section: Piecewise function with compositional sparsity
) = i(i -1 -z) 2 (z -i) 2 on z ∈ [i -1, i] for i ∈ [3]. Although f is smooth within each region, it is only 0-order continuous on [0, 3] × [0, 3].
Warm-up: Piecewise function on E L unit cubes with compositional sparsity. Let the domain be M = [0, E] L , naturally partitioned into E L unit cubes. Consider the following target function:
f (x) = (f 1,i1 (x 1 ), f 2,i2 (x 2 ), • • • , f L,i L (x L )), where i l ∈ {j ∈ [E] : x l ∈ [j -1, j]}, ∀l ∈ [L].
(4)
Here, each subfunction (subtask) f l,i is defined on the interval U l,i = [i -1, i], and f is defined piecewise over
E L regions: U i1 × • • • × U i L , where i l ∈ [E], l ∈ [L]).
The function f in Eq. ( 4) exhibits both:
• Sparsity: each subfucntion f l,i depends only on the coordinate x l , not on the full input x.
• Compositionality: the function f is a hierarchical composition of L selective subfunctions.
Notably, although there are only LE subfunctions, their composition yields E L distinct functions across the domain.
Product manifold. We now extend the above formulation to on general manifolds. Consider a product manifold M = M 1 × M 2 × • • • × M L , where each M l is a compact d l -dimensional manifold in R D , and
L l=1 d l ⩽ D. Each input x ∈ M can be written as x = (x 1 , • • • , x L ) with x l ∈ M l .
By compactness, each submanifold M l admits a finite smooth atlas {(U l,i , ϕ l,i )} i∈[E l ] and an associated partition of unity {ρ li } i∈[E l ] (by Theorem 4.4). Without loss of generality, we can let
E 1 = • • • = E L , denoted by E.
General form: piecewise function on product manifold with compositional sparsity. Consider the target function class admit the formfoot_1 :
f (x) = f out (f 1,i1 (x 1 ), f 2,i2 (x 2 ), • • • , f L,i L (x L )), where i l ∈ {j ∈ [E] : x l ∈ U l,j }, ∀l ∈ [L].(5)
Here, each subfunction (subtask) f l,i is defined on the local subregion U l,i ⊂ M l , and f out composes their outputs. This extends Eq. ( 4) from Euclidean coordinates x l ∈ [0, E] to manifold-based coordinates x l ∈ M l . Since f out can typically be approximated by a dense neural network, we assume f out = id (the identity map) for simplicity.
We now illustrate this formulation with a concrete example.
Table
2: Semantic interpretation of subregions in Example 5.1 M 1 : Math domain M 2 : Language domain U 1,1 : Geometry U 2,1 : English U 1,2 : Algebra U 2,2 : French U 1,3 : Analysis U 2,3 : German Example 5.1. Let M = M 1 × M 2 , where each M l is partitioned into three subregions: M 1 = U 1,1 ∪ U 1,2 ∪ U 1,3 , M 2 = U 2,1 ∪ U 2,2 ∪ U 2,3
, with the interpretations given in Table 2. Each subfunction f 1,i solves the a specific type of math problem (e.g., geometry), while each f 2,i handles text comprehension in a specific language (e.g., English). The full function f defined via Eq. ( 5) encodes 3 × 3 = 9 compositional tasks of the form:
"Understand and solve the [language type] [math type] problem it".
For example, if x 1 ∈ U 1,1 and x 2 ∈ U 2,1 , then f corresponds to the task "understand and solve the English geometry problem".
this section cite: []

Section: Theoretical results and insights
We now present our main theoretical result regarding the expressive power of deep MoE networks for approximating piecewise functions with compositional sparsity. Theorem 5.2 (Main result). Let the target function f be of the form (5), which comprises E L pieces. For each l ∈ [L], assume that the atlas {(U l,i , ϕ l,i )} i∈[E] of M l is regular (Definition 4.7). Then there exists a depth-2L MoE network Ψ ∈ H 2L,E 3,m with m ⩾ Ω E 2 , such that:
∥f -Ψ∥ L∞(M) ⩽ max l∈[L] max i∈[E] E FFN 3,m f l,i ⩽ max l∈[L] max i∈[E] E FFN 2,m f l,i • ϕ -1 l,i approximate a low-dimensional function + max l∈[L] max i∈[E l ] E FFN 2,m ϕ l,i approximate a smooth map f l,i • ϕ -1 l,i C 1 ([0,1] d l ) ⩽ max l∈[L] max i∈[E] Õ m - κ(f l,i ) d l ∧ 1 2 .
Theorem 5.2 establishes that a depth-2L MoE network with E experts per layer can efficiently approximate a piecewise function with E L pieces, provided the function exhibits compositional sparsity. The approximation error consists of two components: (i) approximation of local lowdimensional subfunctions f l,i • ϕ -1 l,i ; (ii) approximation of smooth coordinate maps ϕ l,i . The resulting approximation rate is max l∈[L] max i∈[E] Õ m -κ(f l,i ) d l ∧ 1 2
, which avoids the curse of dimenisonality. In the special case L = 1, this result recovers Theorem 5.2.
this section cite: []

Section: Key insight. The proof (deferred to Appendix B) reveals the following intuitions:
Each pair of MoE layers implements E subtasks, and a depth-2L architecture enables hierarchical composition of E L tasks.
• For each l ∈ [L], the (2l -1)-st and 2l-th layers approximate the E subfunctions f l,i ,(i ∈ [E]) defined on the manifold charts U l,i ∈ M i and assign x l to their correct experts, following the same mechanism as in Theorem 4.8.
• Stacking L such MoE blocks composes these representations hierarchically, approximating the final function
(f 1,i1 (x 1 ), • • • , f L,i L (x L )).
Illustration via Example 5.1. In Example 5.1 with L = 2 and E = 3 (3 subregions for each of math and language). Theorem 5.2 shows that a depth-4 MoE network with 3 experts per layer can express all 3 × 3 = 9 tasks of the form "understand and solve the [language type] [math type] problem". Concretely,
• Experts in Layer 2 (math) implement f 2,i for i = 1, 2, 3, corresponding to "solving geometry/ algebra/analysis problems", respectively.. • Experts in Layer 4 (language) implement f 4,i for i = 1, 2, 3, corresponding to "understanding English/French/German", respectively.
• Layer 1 and the gating in Layer 2 together perform routing for x 1 ; Layer 3 and the gating in Layer 4 together perform routing for x 2 . Theorem 5.3 (Warmup case). Let the target function f be of the form (4). Then there exists a depth-2L MoE network Ψ ∈ H 2L,E 2,m with m ⩾ Ω(E), such that:
∥f -Ψ∥ L∞(M) ⩽ max l∈[L] max i∈[E] E FFN 2,m f l,i approximate a 1-dimensional function ⩽ max l∈[L] max i∈[E] O m -κ(f l,i )∧ 1 2 .
Compared to Theorem 5.2, this result requires only: (i) shallower expert networks (2-layer instead of 3-layer), (ii) smaller expert width (m ⩾ Ω(E) instead of Ω(E 2 )) due to the simple geometry (Euclidean cubes).
Although each subfunction may have high-order smoothness (κ(f l,i ) ≫ 1), the composite function f can exhibit only low-order smoothness (e.g., κ(f ) = 0, 1) due to low-order regularity at the interfaces between adjacent regions. As a result, directly approximating the global function f is inefficient. A natural and efficient approach is to decompose the approximation problem into localized subproblems, each defined on a simple subregion with high regularity. Our constructive MoE networks can provably achieve this approach.
this section cite: []

Section: Experimental Validation
To support our main theoretical results, we conduct two new experiments, each aligned with one of our key insights. The experimental details are shown in Appendix C.
this section cite: []

Section: Experiment I. Shallow MoEs for low-dimensional functions.
To validate our theoretical insight in Section 4 (Theorem 4.8): shallow MoE networks can efficiently approximate functions supported on low-dimensional manifolds and overcome the curse of dimensionality.
Specifically, we consider the low-dimensional manifold M = {x ∈ R D :
x 2 1 + x 2 2 = 1; x i = 0, ∀i > 2} embedded in R D with D > 2.
The target function is f (x) = sin(5x 1 ) + cos(3x 2 ), defined on M. As a model, we consider "1-4-MoE", a 1-layer MoE comprising 1 router and 4 experts, where each expert is a two-layer ReLU network with hidden width 10. To validate whether MoE can overcome the curse of dimensionality, we vary the input dimension D ∈ {16, 32, 64, 128}.
As shown in Table 3, one can see that: as D increases, the test error of MoE does not increase significantly and remains stable. This supports our insight that shallow MoEs efficiently approximate functions on low-dimensional manifolds and avoid the curse of dimensionality. As defined in our Figure 3, we consider the piecewise function f with compositional sparsity defined over 3 2 = 9 unit cubes. As the model, we consider "2-3-MoE" (a 2-layer MoE comprising 2 routing layers and 2 expert layers with 3 experts each); To illustrate the role of depth, we also consider a shallow "1-6-MoE", with comparable parameter count. Each expert is a two-layer ReLU FFN with hidden width m ∈ {16, 32, 64, 128}. To validate whether 2-3-MoE and 1-6-MoE can approximate this target, we vary the hidden width m.
The results, shown in Table 4, illustrate that: (i) As m increases, 2-3-MoE achieves rapidly decreasing error. This supports that the depth-2 MoE with 3 experts per layers can efficiently approximate this piecewise function with 3 2 distinct pieces; (ii) In contrast, 1-6-MoE exhibits a performance plateau, revealing its limited expressive power. This highlights the crucial role of depth in modeling such compositional structures.
this section cite: []

Section: Conclusion and Future Work
In this work, we provide a theoretical study of the expressive power of MoE networks for modeling structured complex tasks. For shallow MoE networks, we show that they can efficiently approximate functions supported on low-dimensional manifolds, overcoming the curse of dimensionality. For deep MoE networks, we establish that, when the target exhibits a compositional sparsity structure, they can approximate piecewise functions consisting of exponentially many distinct pieces, effectively modeling an exponential number of tasks.
Beyond compositional sparsity. Our analysis focuses on compositional sparsity, a structure that is both theoretically rich and practically relevant. However, real-world problems may involve other forms of sparsity, such as group sparsity, graph sparsity, or temporal sparsity. Characterizing the expressive power of deep MoE networks under these alternative structures is an important direction for future work.
this section cite: []

Section: References
Ref_id:b0 Title: Phi-3 technical report: A highly capable language model locally on your phone Year: (2024)
Ref_id:b1 Title: Quadratic gating functions in mixture of experts: A statistical insight Year: (2024)
Ref_id:b2 Title: Breaking the curse of dimensionality with convex neural networks Year: (2017)
Ref_id:b3 Title: A theoretical view on sparsely activated networks Year: (2022)
Ref_id:b4 Title: Laplacian eigenmaps for dimensionality reduction and data representation Year: (2003)
Ref_id:b5 Title: Dynamic programming Year: (1966)
Ref_id:b6 Title: A survey on mixture of experts in large language models Year: (2025)
Ref_id:b7 Title: An introduction to compressive sampling Year: (2008)
Ref_id:b8 Title: Efficient approximation of deep relu networks for functions on low dimensional manifolds Year: (2019)
Ref_id:b9 Title: Towards understanding mixture of experts in deep learning Year: (2022)
Ref_id:b10 Title: On the benefits of learning to route in mixture-of-experts models Year: (2023)
Ref_id:b11 Title: Compressed sensing Year: (2006)
Ref_id:b12 Title: Glam: Efficient scaling of language models with mixture-of-experts Year: (2022)
Ref_id:b13 Title: Image denoising via sparse and redundant representations over learned dictionaries Year: (2006)
Ref_id:b14 Title: Curvature measures Year: (1959)
Ref_id:b15 Title: Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity Year: (2022)
Ref_id:b16 Title: Reducing the dimensionality of data with neural networks Year: (2006)
Ref_id:b17 Title: Adaptive mixtures of local experts Year: (1991)
Ref_id:b18 Title: Hierarchical mixtures of experts and the em algorithm Year: (1994)
Ref_id:b19 Title: Auto-encoding variational bayes Year: (2013)
Ref_id:b20 Title: Mixture of experts meets prompt-based continual learning Year: (2024)
Ref_id:b21 Title: Gshard: Scaling giant models with conditional computation and automatic sharding Year: (2020)
Ref_id:b22 Title: Theory on mixture-of-experts in continual learning Year: (2024)
Ref_id:b23 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b24 Title: Gating dropout: Communicationefficient regularization for sparsely activated transformers Year: ()
Ref_id:b25 Title: Rates of approximation by relu shallow neural networks Year: (2023)
Ref_id:b26 Title: Deep vs. shallow networks: An approximation theory perspective Year: (2016)
Ref_id:b27 Title: A comprehensive survey of mixture-of-experts: Algorithms, theory, and applications Year: (2025)
Ref_id:b28 Title: Statistical advantages of perturbing cosine router in sparse mixture of experts Year: (2024)
Ref_id:b29 Title: Finding the homology of submanifolds with high confidence from random samples Year: (2008)
Ref_id:b30 Title: Emergence of simple-cell receptive field properties by learning a sparse code for natural images Year: (1996)
Ref_id:b31 Title: How deep sparse networks avoid the curse of dimensionality: Efficiently computable functions are compositionally sparse Year: (2023)
Ref_id:b32 Title: Nonlinear dimensionality reduction by locally linear embedding Year: (2000)
Ref_id:b33 Title: Provable approximation properties for deep neural networks Year: (2018)
Ref_id:b34 Title: A global geometric framework for nonlinear dimensionality reduction Year: (2000)
Ref_id:b35 Title: Regression shrinkage and selection via the lasso Year: (1996)
Ref_id:b36 Title: Attention is all you need Year: (2017)
Ref_id:b37 Title: Understanding the expressive power and mechanisms of transformer for sequence modeling Year: (2024)
Ref_id:b38 Title: Qwen2 technical report Year: (2024-03)
Ref_id:b39 Title: Qwen2. 5 technical report Year: (2006)
Ref_id:b40 Title: Optimal rates of approximation by shallow relu k neural networks and applications to nonparametric regression Year: (2024)
Ref_id:b41 Title: Deep network approximation: Beyond relu to diverse activation functions Year: (2024)
Ref_id:b42 Title: Taming sparsely activated transformer with stochastic experts Year: (2021)
