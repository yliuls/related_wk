Title: Exogenous Isomorphism for Counterfactual Identifiability
Abstract: This paper investigates ∼ L3 -identifiability, a form of complete counterfactual identifiability within the Pearl Causal Hierarchy (PCH) framework, ensuring that all Structural Causal Models (SCMs) satisfying the given assumptions provide consistent answers to all causal questions. To simplify this problem, we introduce exogenous isomorphism and propose ∼ EI -identifiability, reflecting the strength of model identifiability required for ∼ L3 -identifiability. We explore sufficient assumptions for achieving ∼ EI -identifiability in two special classes of SCMs: Bijective SCMs (BSCMs), based on counterfactual transport, and Triangular Monotonic SCMs (TM-SCMs), which extend ∼ L2 -identifiability. Our results unify and generalize existing theories, providing theoretical guarantees for practical applications. Finally, we leverage neural TM-SCMs to address the consistency problem in counterfactual reasoning, with experiments validating both the effectiveness of our method and the correctness of the theory.

Section: Introduction
The purpose of counterfactual reasoning is to answer questions about hypothetical, unobserved worlds. It has been applied in tasks such as fairness evaluation (Kusner et al., 2017), explanation generation (Karimi et al., 2020), harm quantification (Richens et al., 2022), and policy optimization (Tsirtsis & Rodriguez, 2023). Counterfactual identification is a subtask of counterfactual reasoning, aiming to determine whether all models satisfying certain reasonable assumptions yield the same answer to a specific counterfactual question. It is critical to ensure the dependability of counterfactual reasoning results, as the reasoning outcomes can only be consistent with the assumptions -and Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
thus confirm the reliability of the constructed model -if counterfactual identifiability is guaranteed. (Pearl & Mackenzie, 2018) categorized causal questions into three levels of human cognition, termed the Pearl Causal Hierarchy (PCH) (Bareinboim et al., 2022). Counterfactual reasoning corresponds to human imagination, situated at the highest level L 3 of the PCH, encoding the most intricate and nuanced information. Structural Causal Models (SCMs) provide specific semantics for addressing counterfactual questions. Counterfactual identifiability on SCMs guarantees that all SCMs adhering to the assumptions yield consistent results for L 3 quantities. Prior research has constrained causal structures to identify specific counterfactual effects (Shpitser & Pearl, 2008;Correa et al., 2021;Xia et al., 2023), or limited causal mechanisms to determine counterfactual outcomes (Lu et al., 2020;Nasr-Esfahany et al., 2023;Scetbon et al., 2024). Further studies investigate model identifiability for counterfactuals, offering empirical evidence that counterfactual outcomes are identifiable (Khemakhem et al., 2021;Javaloy et al., 2023).
This work introduces a novel identification target: identification across the entire counterfactual layer of the PCH. This requires that all SCMs satisfying the assumptions yield consistent results for any counterfactual statement. Within the PCH, since the counterfactual layer L 3 encodes all causal information, the identifiability of the SCMs in L 3 implies that these SCMs provide consistent answers to all causal questions, rendering them indistinguishable for any causal statements. Thus, identifiability over the counterfactual layer represents the most stringent and comprehensive goal for causal quantity identifiability within the PCH.
To achieve this goal, we first examine the identifiability problem in causal inference from the perspective of model identifiability in Section 2. In this context, identifiability over the counterfactual layer is denoted as ∼ L3 -identifiability. To simplify the ∼ L3 -identifiability problem, we establish an alternative form of identifiability in Section 3, denoted as ∼ EIidentifiability, which is induced by an equivalence relation termed exogenous isomorphism. This form of identifiability demonstrates that fully recovering the exogenous variables of SCMs is unnecessary to ensure consistent results for any counterfactual quantities, clarifying the strength of assumptions required for counterfactual layer identification.
We then explore sets of assumptions that induce ∼ EIidentifiability, focusing on two special classes of SCMs. The first class, termed Bijective SCMs (BSCMs), is studied in Section 4, where we induce exogenous isomorphism between BSCMs from the perspective of counterfactual transport (Theorem 4.6), offering a novel interpretation for counterfactual identifiability. The second class, termed Triangular Monotonic SCMs (TM-SCMs), is examined in Section 5. We identify a simple method to induce exogenous isomorphism between TM-SCMs (Corollary 5.4), which can be viewed as a strengthened version of ∼ L2 -identifiability for Markovian Causal Bayesian Networks. This corollary unifies and generalizes prior theories for proving counterfactual outcome identifiability, indirectly demonstrating that several models used in previous works for counterfactual outcome identifiability are theoretically ∼ L3 -identifiable. This provides theoretical guarantees for their safe application in practice. Finally, leveraging this theory, we utilize TM-SCMs parameterized by neural networks in Section 6 to address counterfactual identification and estimation problems. Experimental results in Section 8 on these models empirically support the validity of our theory.
this section cite: ['b26', 'b21', 'b36', 'b47', 'b33', 'b44', 'b9', 'b54', 'b28', 'b29', 'b41', 'b23', 'b20']

Section: Preliminaries
In this section, we provide the background for understanding the problem and describe the problem setting. We begin by reviewing the relevant concepts of SCMs. In Section 2.1, we present the definition of SCM and introduce key notions such as do-intervention and causal order. Subsequently, in Section 2.2, we leverage these concepts to logically characterize L 1 , L 2 , L 3 in PCH and define L 3 -consistency. Next, in Section 2.3, we introduce counterfactual identification problem. Finally, in Section 2.4, we elaborate on model identifiability, establish the connection between causal identifiability and model identifiability, and motivate the research objective of this work: ∼ L3 -identifiability.
this section cite: []

Section: Notation
We study the problem from a more general measure-theoretic perspective. Let the probability space be (Ω, F, P ), where the set Ω is the sample space, the σalgebra F on Ω is the event space, and P is the probability measure. We assume all measurable spaces are standard Borel. We denote by uppercase X a random variable taking values from the measurable space (Ω X , F X ), which is a measurable function from Ω to Ω X , where Ω X is also called the domain of X. For a given finite index set I, we use uppercase bold X to denote a collection of random variables (X i ) i∈I with the associated space (Ω X , F X ), where Ω X = i∈I Ω Xi and F X = i∈I F Xi are the product space and the product σ-algebra, respectively. For a subset of indices I ⊆ I, (X i ) i∈I is abbreviated as X I or renamed as Y with Y ⊆ X, where the corresponding index set is denoted by I Y . The distribution of X is represented by P X , defined as the pushforward of P under X, such that P X = X ♯ P . A set X ∈ F X represents a random event, with the associated probability term P(X ∈ X ) = P X (X ).
this section cite: []

Section: Structural Causal Model
Definition 2.1 (Structural Causal Model (SCM)). An SCM is a tuple M = ⟨I, Ω V , Ω U , f , P U ⟩, where I is a finite index set. The product space Ω V = i∈I Ω Vi and Ω U = i∈I Ω Ui denote the domain of endogenous and exogenous variables, respectively. The measurable function f : Ω V × Ω U → Ω V = (f i ) i∈I specifies the causal mechanisms, where each f i : Ω V pa(i) × Ω Ui → Ω Vi is associated with a set of indices pa(i) ⊆ I. The probability measure P U is called the exogenous distribution. If P U = i∈I P Ui is a product measure, the SCM is said to be Markovian.
In this work, we primarily focus on recursive SCMs, which are defined such that a partial order ⪯ exists on the index set I, and for any pair of indices i, j ∈ I, if j ⪯ i, then i / ∈ pa(j). If the partial order ⪯ admits a linear extension ≤, we call it the causal order of the SCM, which can be constructed using a topological sorting algorithm and is not necessarily unique. Recursiveness ensure that the solution of the SCM exists and is unique. According to (Bongers et al., 2016), if an SCM has a unique solution, then there exists a function Γ such that for almost all u ∈ Ω U and v ∈ Ω V , v = Γ(u) if and only if v = f (v, u). We refer to Γ as the solution mapping of the recursive SCM.
SCMs also allow for operations called do-interventions. For a subset of indices I X ⊆ I, performing an intervention to set X = V I X to a specific value x corresponds to deriving a new SCM M [x] = ⟨I, Ω V , Ω U , f [x] , P U ⟩, called a submodel. In M [x] , f [x] = (f i[x] ) i∈I is defined such that
f i[x] = x i i ∈ I X , f i i ∈ I \ I X .
Endogenous variables in M [x] will be denoted as V [x] .
Another property of recursive SCMs is that any submodel is also a recursive SCM. This implies that, in the submodel, given a value u, the endogenous variables V [x] are determined as Γ [x] (u). Under this premise, the value of the endogenous variables Y [x] under a given u is represented as Y M [x] (u) = (Γ [x] (u)) I Y , called the potential response.
2.2. Pearl Causal Hierarchy (Pearl & Mackenzie, 2018) categorized causal questions into three levels of human cognition: seeing, doing, and imagining. To formalize these questions and delineate the logical differences between the three levels, symbolic languages L 1 , L 2 , L 3 are introduced. Each language consists of L istatements φ, where every φ ∈ L i is a Boolean combination of inequalities between polynomials over probability terms P(α i ). For L 1 , P(α 1 ) takes the form P(Y ∈ Y), representing the probability of Y occurring, making L 1 a standard probabilistic logic. L 2 introduces interventions, with P(α 2 ) in the form P(Y [x] ∈ Y), denoting the probability of Y occurring when X is intervened to x. L 3 further encodes conjunctions under different interventions, making P(α 3 ) take the form P(Y * ∈ Y * ), where Y * = (Y j[xj ] ) j∈1:n represents the collection of endogenous variables Y j[xj ] from n submodels M [x1] , M [x2] , . . . , M [xn] . Thus, Y * ∈ Y * is logically equivalent to the conjunction j∈1:n Y j[xj ] ∈ Y j .
An SCM M provides semantics for the symbolic languages L 1 , L 2 , L 3 . For L 3 , any term of the form P(Y * ∈ Y * ) can be evaluated by M as P M Y * (Y * ), where
P M Y * is called the counterfactual distribution. Let Y M * = (Y M j[x j ] ) j∈1:n . Since each Y j[xj ] = Y M j[x j ] • U almost surely, the P M Y *
is the pushforward measure (Y M * ) ♯ P U , yielding:
P M Y * (Y * ) = P U (Y M * ) -1 [Y * ] .(1)
Equation 1 is called L 3 -valuation. After the L 3 -valuation, a statement φ ∈ L 3 evaluated by M, denoted as φ(M), can be checked for validity. If φ(M) holds, we write M |= φ.
The Pearl Causal Hierarchy (PCH) is defined as the combination of the above syntax and semantics. Specifically, when an SCM M is given, the collection of observational, interventional, and counterfactual distributions defined syntactically by L 1 , L 2 , L 3 and semantically by L 1 , L 2 , L 3valuations constitutes the PCH. In this sense, the PCH encapsulates all causal information entailed by an SCM.
Furthermore, the PCH exhibits a strict hierarchy. In terms of syntax, the representations of the terms P(α i ) imply that L 1 ⊊ L 2 ⊊ L 3 . In terms of logical expressiveness, similar theorems have been established, known as the Causal Hierarchy Theorem (CHT) (Bareinboim et al., 2022).
To illustrate the expressiveness of L i , consistency is introduced. Let L i (M) = {φ(M) | φ ∈ L i } denote the L i -theory of M, then L i -consistency is defined as follows:
Definition 2.2 (L i -Consistency). SCMs M (1) and M (2) are L i -consistent, denoted M (1) ∼ Li M (2) , if their L itheories are identical, i.e., L i (M (1) ) = L i (M (2) ).
The CHT demonstrates that in the PCH, ∼ Li generally does not imply ∼ Lj for i < j, formally indicating that higher levels possess greater expressiveness than lower levels.
this section cite: ['b3', 'b33']

Section: Counterfactual Identification
In practice, the underlying true SCM M * is almost never fully specified, making direct reasoning on M * impractical. Consequently, researchers rely on partial knowledge about M * to address causal questions, which necessitates constructing a proxy M ′ that satisfies this partial knowledge and performing equivalent reasoning on M ′ . Causal identification is the task of proving or reasoning whether any proxy constructed from partial knowledge provides consistent answers to specific causal questions.
For example, when the true SCM M * is assumed to be Markovian and both observational distribution and the causal graph are available, constructing a Markovian Causal Bayesian Network (CBN) suffices to answer any causal question in L 2 via truncated factorization (Pearl, 2009). Thus, Markovian CBNs can be regarded as identifiable at the intervention level. When the Markovian assumption fails, semi-Markovian CBNs can still answer a subset of causal questions in L 2 using do-calculus.
Regarding counterfactuals, as they represent the highest level in the PCH, achieving counterfactual identifiability requires stricter conditions and more complex methodologies. For instance, even Markovian CBNs cannot ensure consistency when answering certain L 3 questions (Nasr-Esfahany & Kiciman, 2023). Prior research on counterfactual identification often focuses on identifying a subset of counterfactual statements φ φ φ ⊆ L 3 . These efforts can be categorized into constraining causal structures to identify counterfactual effects, such as (Shpitser & Pearl, 2008), or constraining causal mechanisms to identify counterfactual outcomes, such as (Nasr-Esfahany et al., 2023).
this section cite: ['b32', 'b29', 'b44']

Section: Model Identifiability
SCMs are a special class of generative models, where the exogenous distribution and causal mechanisms can be interpreted as the latent distribution and generative function of the model, respectively. Identifiability has been a continuously discussed topic in the literature, appearing in representation learning (Bengio et al., 2013), causal discovery (Glymour et al., 2019), causal representation learning (Schölkopf et al., 2021), and causal quantity inference (Pearl, 2009). Providing a broad definition of identifiability may lead to ambiguity. Borrowing from the definition of identifiability in (Khemakhem et al., 2020), these notions of identifiability can be unified through equivalence classes:
Definition 2.3 (∼-Identifiability). Let A denote a set of assumptions, [A] = {M | M |= A} includes all models satisfying the assumptions, and ∼ denote an equivalence relation between models. Then a model is said to be ∼identifiable from A, or identifiable up to ∼-equivalence class, if for any M (1) , M (2) ∈ [A], M (1) ∼ M (2) .
We now interpret the problem of causal identifiability from the perspective of ∼-identifiability. Suppose the causal question involves a set of L i -statements φ φ φ ⊆ L i , and two SCMs M (1) and M (2) are said to be φ φ φ-consistent if φ φ φ(M (1) ) = φ φ φ(M (2) ). Since φ φ φ-consistency is an equiva-lence relation between SCMs, denoted as M (1) ∼ φ φ φ M (2) , the problem of causal identification with respect to φ φ φ can be reframed as a problem of ∼ φ φ φ -identifiability.
The CHT states that when A contains only low-level knowledge, ∼ φ φ φ -identifiability is often unattainable. Achieving this requires assuming higher-level information. For instance, under the assumptions of known L 1 observational distribution, causal graph, and Markovianity, the causal graph encodes structural constraints on L 2 , enabling the constructed CBN to answer all questions in L 2 . Since the union of all statements φ φ φ ⊆ L 2 equals L 2 , this property is also referred to as ∼ L2 -identifiability.
This paper focuses on ∼ L3 -identifiability, which is an enhanced version of ∼ L2 -identifiability, requiring ∼ φ φ φidentifiability for any φ φ φ ⊆ L 3 . If A satisfies ∼ L3identifiability, then by the definition of L 3 -consistency, any M (1) , M (2) ∈ [A] satisfy M (1) ∼ L3 M (2) . Since L 3 represents the highest level in the PCH and encodes all causal information of the SCM, if M (1) ∼ L3 M (2) , the two models are indistinguishable under any causal statement. Therefore, ∼ L3 -identifiability is the ultimate goal for causal identifiability within the PCH.
this section cite: ['b2', 'b42', 'b32', 'b22']

Section: Exogenous Isomorphism
∼ L3 -identifiability, by definition, is not straightforward to handle, as it is indirectly defined through the PCH rather than directly based on the model itself. Therefore, we aim to find a simpler model-based identifiability notion that implies ∼ L3 -identifiability, thereby simplifying the problem.
One possible choice is =-identifiability, which requires uniquely identifying the generative model. Some prior works have achieved =-identifiability (Xi & Bloem-Reddy, 2023), which indirectly induces ∼ L3 -identifiability. This serves as one approach to addressing ∼ L3 -identifiability, but the required assumptions are often too strong. An alternative property is defined via counterfactual equivalence (Peters et al., 2017, Proposition 6.49), which-while not insisting on uniqueness of the causal mechanisms-does demand full recovery of the exogenous distribution.
Between =-identifiability, counterfactual equivalence and ∼ L3 -identifiability, there should exist other forms of ∼identifiability, as counterfactual reasoning does not require a completely fixed latent representation. This is akin to humans being able to answer "what-if" questions without fully understanding the underlying factors of the physical world.
Identifying such a form of ∼-identifiability is a goal worth exploring. Motivated by this, we have identified an equivalence relation between SCMs, denoted as ∼ EI and referred to as exogenous isomorphism, for which ∼ EI -identifiability strictly implies ∼ L3 -identifiability, yet it is weaker than the previous two. This characterization more precisely reflects the strength of model identifiability required to achieve complete counterfactual identifiability.
Let the recursive SCMs M (k) = ⟨I, Ω V , Ω
U , f (k) , P
U ⟩ share the index set I and the domain of endogenous variables Ω V . For a given endogenous value v ∈ Ω V , we abbreviate f (k)
i (v pa (k) (i) , •) as f (k) i (v, •).
Exogenous isomorphism is then defined as:
Definition 3.1 (Exogenous Isomorphism). Recursive SCMs M (1) and M (2) are said to be exogenously isomorphic, denoted M (1) ∼ EI M (2) , if there exists a shared causal ordering ≤ and function h : Ω (1)
U → Ω (2) U satisfying: • Component-wise Bijection: For each i ∈ I, h = (h i ) i∈I ,
where
h i : Ω (1) Ui → Ω (2) Ui is a bijection; • Exogenous Distribution Isomorphism: P (2) U = h ♯ P (1) U ; • Causal Mechanism Isomorphism: For each i ∈ I, for almost every u (1) i ∈ Ω (1) Ui and all v ∈ Ω V , f(2)
i (v, h i (u (1) i )) = f (1) i (v, u(1)
i ). The implication of ∼ EI -identifiability for ∼ L3 -identifiability is formally stated in the following theorem:
Theorem 3.2 (∼ EI Implies ∼ L3 ). For recursive SCMs M (1) and M (2) , if M (1) ∼ EI M (2) , then M (1) ∼ L3 M (2) .
this section cite: ['b52', 'b34']

Section: Sketch of proof
The mechanism isomorphism of each f i can be progressively deduced into the potential responses, ensuring V
(1)
M * = V (2) M * •h. Given P (2) U = h ♯ P (1) U , we have (V (2) M * • h) ♯ P (1) U = (V (2) M * ) ♯ (h ♯ P (1) U ), i.e., (V (1) M * ) ♯ P (1) U = (V (2) M * ) ♯ P (2) U .
Thus, all L 3 -evaluations for M (1) and M (2) are identical, and any statement φ ∈ L 3 yields the same result. Therefore, M (1) ∼ L3 M (2) . For a detailed proof, see Appendix A.2.
this section cite: []

Section: Bijective SCM
Once identifiability is defined, the next step is to explore the assumption sets that induce identifiability. In this work, for simplicity, we target some special settings of recursive SCMs. Throughout, the set of positive integers {i | l ≤ i ≤ r, i ∈ N + } is abbreviated as l : r for l, r ∈ N + . Moreover, we assume that for each i ∈ I, the domain of exogenous variables Ω Ui and the domain of endogenous variables Ω Vi are both indexed R di with 1 : d i as the index set. That is, both exogenous and endogenous variables are random vectors, and the nested index set I = {(i, j) | i ∈ I, j ∈ 1 : d i } indexes the individual dimensions of Ω U and Ω V .
Since the cardinalities of the exogenous and endogenous domains are equal, a bijection can be established between these domains. A bijection implies no information loss. If the causal mechanism is also bijective, then for any observation in the endogenous domain, a distinguishable latent encoding can be identified in the exogenous domain. Below, we formally define this specific setting of SCMs: Definition 4.1 (Bijective SCM (BSCM)). A recursive SCM M is called a bijective SCM (BSCM) if its solution mapping Γ is a bijection.
Proposition 4.2. A recursive SCM M is a BSCM if and only if f i (v, •) is a bijection for every i ∈ I and all v ∈ Ω V .
For BSCMs M (foot_1) and M (2) , since their solution mappings
Γ (k) : Ω (k) U → Ω V are bijections, it is evident that the composition (Γ (2) ) -1 • Γ (1) : Ω (1) U → Ω (2)
U is also a bijection. Given that exogenous isomorphism requires a bijection between Ω (1) U and Ω
(2) U , a natural question arises: can (Γ (2) ) -1 •Γ (1) directly serve as an exogenous isomorphism?
The following theorem provides a precise answer.
this section cite: []

Section: Theorem 4.3 (BSCM-EI).
If two BSCMs M (1) and M (2) share a common causal order ≤ and the same observational distribution P V , then M (1) ∼ EI M (2) if and only if for every i ∈ I, there exists a bijection h i : Ω (1)
Ui → Ω (2) Ui such that for all v ∈ Ω V , (f (2) i (v, •)) -1 • (f (1) i (v, •)) = h i almost surely. 1
this section cite: []

Section: ∼ EI -Identification for BSCM
To derive assumption sets that imply ∼ EI -identifiability, we need to restrict our perspective to a single SCM M. Consider the conditions in Theorem 4.3, where there exists a function h i : Ω (1)
Ui → Ω (2) Ui such that for all v ∈ Ω V , (f (2) i (v, •)) -1 • (f (1) i (v, •)) = h i holds almost surely. Now, consider different v, v ′ ∈ Ω V . Since both f (1) i (v, •) and f (2) i (v, •) are bijections, by the associativity of composition, (f (1) i (v ′ , •))•(f (1) i (v, •)) -1 = (f (2) i (v ′ , •))•(f (2) i (v, •)) -1
almost surely, where both sides come from the SCM M, and we explicitly name this concept as follows:
Definition 4.4 (Counterfactual Transport). For a BSCM M, the function K M : Ω V × Ω V × Ω V → Ω V is called the counterfactual transport if K M = (K M,i ) i∈I and for every i ∈ I and all v, v ′ ∈ Ω V , the component K M,i (•, v, v ′ ) = (f i (v ′ , •)) • (f i (v, •)) -1 .
Under Markovianity, the practical meaning of counterfactual transport is the transport between conditional distributions: Proposition 4.5. If the BSCM M is Markovian, then for almost all v, v ′ ∈ Ω V , the conditional distributions satisfy
P Vi|V pa(i) (•, v ′ ) = (K M,i (•, v, v ′ )) ♯ P Vi|V pa(i) (•, v). 2
Combining this with Theorem 3.2, we find that when all counterfactual transports are fixed, ∼ EI -identifiability can be achieved. Let the following assumptions be defined: (i) A BSCM : M is a BSCM; (ii) A ≤ : the total order ≤ is a causal order for M; (iii) A P V : the observational distribution of M is P V with strictly positive density; (iv) A K : there exists a function K :
Ω V × Ω V × Ω V → Ω V = (K i ) i∈I such that counterfactual transport K M = K almost surely. Let A {BSCM,≤,P V ,K} = {A BSCM , A ≤ , A P V , A K }. Then: Theorem 4.6 (EI-ID from Counterfactual Transport). An SCM is ∼ EI -identifiable from A {BSCM,≤,P V ,K} .
Verifying the assumption set A K in practice is challenging, making Theorem 4.6 difficult to apply. By combining Proposition 4.5 and considering Markovian BSCMs, a natural direction is to identify specific types of transport that are always well-defined between conditional distributions. One such transport, the KR transport, is constructed using one-dimensional conditional cumulative density functions in 1 : d order, ensuring that the mass at each point of the distribution follows a lexicographical order, and: Lemma 4.7 (Santambrogio 2015, Proposition 2.18). Given any two distributions P and P ′ on R d with strictly positive densities, the KR transport T : R d → R d such that P = T ♯ P ′ always exists and is almost surely unique.
The KR transport induces a special case of A {BSCM,≤,P V ,K} in Theorem 4.6. Let the following assumptions be defined: (i) A M : M is Markovian; (ii) A KR : for each i ∈ I, the counterfactual transport component K M,i is almost surely equivalent to the KR transport.
Let A {BSCM,≤,M,P V ,KR} = {A BSCM , A ≤ , A M , A P V , A KR }. Then: Theorem 4.8 (EI-ID from KR Transport). An SCM is ∼ EI - identifiable from A {BSCM,≤,M,P V ,KR} .
this section cite: []

Section: Triangular Monotonic SCM
We call a function T(x) = (T j (x 1:j-1 , x j )) j∈1:d , where T : R d → R d , a triangular mapping if the j-th component T j depends only on x 1:j . The name originates from the fact that the Jacobian matrix of such a mapping is triangular. For a triangular mapping T, we define its monotonicity signature ξ(T) = (ξ(T j )) j∈1:d as:
ξ(T j ) =      1 ∀x 1:j-1 ∈ R j-1 , T j (x 1:j-1 , •) is s.m.i. -1 ∀x 1:j-1 ∈ R j-1 , T j (x 1:j-1 , •) is s.m.d. 0 otherwise ,
where s.m.i. and s.m.d. abbreviate strictly monotonically increasing and decreasing, respectively. If d = d j=1 ξ(T j ), i.e., every component is consistently s.m.i., we call T a triangular monotonic increasing (TMI) mapping. TMI mappings are closely related to KR transport: Lemma 5.1 (Jaini et al. 2019, Theorem 1). Given any two distributions P and P ′ on R d with strictly positive densities, if a TMI mapping T : R d → R d satisfies P = T ♯ P ′ , then T is almost surely equivalent to the KR transport.
TMI mappings can be further generalized. For a triangular mapping T, if d = d j=1 |ξ(T j )|, i.e., every component is either consistently s.m.i. or s.m.d., we call T a triangular monotonic (TM) mapping. TM mappings, in addition to being bijective due to monotonicity, exhibit special properties: they remain TM mappings under inversion, composition, and selection of contiguous components. Moreover, their monotonicity signatures adhere to specific rules, which are elaborated in Appendix A.4. Crucially, for two TM mappings T (1) and T (2) with identical monotonicity signatures, T (2) • (T (1) ) -1 is always a TMI mapping.
Next, we aim to relate the solution mapping Γ of an SCM to TM mappings. However, TM mappings are defined over the index set 1 : d, whereas SCMs are defined over the nested index set I = {(i, j) | i ∈ I, j ∈ 1 : d i }. To bridge this gap, we introduce the concept of flattening, a generalization of permutation. For a finite index set I, we call a bijection ι : I → 1 : |I| a flattening mapping of I, and flattening refers to re-indexing according to ι. Define the re-indexing
P ι : i∈I Ω Xi → |I| i=1 Ω Xi such that for every i ∈ I, P ι ((x i ) i∈I ) = (x ι -1 (j) ) j∈ι[I] ,
where ι[I] is the image set. Since ι is bijective, P ι is also bijective, and for any J ⊆ 1 : |I|, we have
(P -1 ι )((x j ) j∈J ) = (x ι(i) ) i∈ι -1 [J]
, where ι -1 [I] is the pre-image set.
For the nested index set I and a total order ≤ on I, if a flattening mapping ι satisfies ι(i, j) -ι(i, j ′ ) = j -j ′ for any (i, j), (i, j ′ ) ∈ I and ι(i, j) < ι(i ′ , j ′ ) for any (i, j), (i ′ , j ′ ) ∈ I with i < i ′ , we call it a vectorization under ≤. In other words, vectorization merges all random vectors into a unified order. Based on whether the solution mapping Γ is a TM map under vectorization, we define a more specialized class of SCMs:
this section cite: []

Section: Definition 5.2 (Triangular Monotonic SCM (TM-SCM)).
A recursive SCM M is called a triangular monotonic SCM (TM-SCM) if there exists a vectorization ι under a causal order such that the re-indexed P ι • Γ is a TM mapping.
Proposition 5.3. A recursive SCM M is a TM-SCM if and only if f i (v, •) is a TM mapping and there exists ξ i such that ξ(f i (v, •)) = ξ i for every i ∈ I and all v ∈ Ω V .
this section cite: []

Section: ∼ EI -Identification for TM-SCM
As a subclass of BSCMs, TM-SCMs allow the assumption set in Theorem 4.8 to be further simplified. The new identifiability strategy is specifically tied to TM-SCMs. Let A {TM-SCM,≤,M,P V } = {A TM-SCM , A ≤ , A M , A P V }, where A TM-SCM states that M is a TM-SCM. Then:
Corollary 5.4 (EI-ID from Triangular Monotonicity). An SCM is ∼ EI -identifiable from A {TM-SCM,≤,M,P V } .
Sketch of Proof By combining the definition of TM-SCMs with the properties of TM mappings, any counterfactual transport in a TM-SCM is a TMI mapping. Furthermore, under Markovianity, given the causal order ≤ and observational distribution P V , the KR transport is almost uniquely determined by Lemmas 4.7 and 5.1, which then implies Theorem 4.8. A detailed proof is provided in Appendix A.4. Corollary 5.4 (together with Theorem 3.2) can be viewed as a strengthened version for Markovian CBNs, which exhibits ∼ L2 -identifiability, i.e., SCMs are ∼ L2 -identifiable from A {≤,M,P V } . Corollary 5.4 augments A {≤,M,P V } with the additional assumption A TM-SCM , strengthening ∼ L2identifiability to ∼ L3 -identifiability.
Moreover, this extends and unifies the theoretical results in (Lu et al., 2020, Theorem 1), (Nasr-Esfahany et al., 2023, Theorem 5.1), and (Scetbon et al., 2024, Theorem 2.14). Specifically, these theorems focus on counterfactual identifiability from the perspectives of bijectivity, monotonicity, and Markovianity, demonstrating identifiability based on abduction-action-prediction. Compared to these results, our theoretical contributions offer the following advancements: (i) Generalizing the measurable space of each endogenous variable from a scalar R to a vector R di , enabling support for a broader class of SCMs; (ii) Demonstrating ∼ EI -identifiability based on exogenous isomorphism theory (i.e. Theorem 3.2), which implies ∼ L3 -identifiability rather than merely the identifiability of counterfactual outcomes.
6. Neural TM-SCM Problem formulation Consider the true underlying SCM M * = ⟨I, Ω V , Ω * U , f * , P * U ⟩, where Ω * U , f * , and P * U are unknown. Assuming M * is a Markovian TM-SCM with causal order ≤, how can one construct a parameterized proxy SCM M θ = ⟨I, Ω V , Ω U θ , f θ , P U θ ⟩ based on the observational dataset
D V = {v i | v (i) ∼ P V } N
i=1 such that M θ is as L 3 -consistent with M * as possible?
According to Corollary 5.4 and Theorem 3.2, this problem is equivalent to constructing M θ such that it satisfies A {TM-SCM,≤,M,P V } as closely as possible. This entails four construction objectives: (i) G1: M θ |= A TM-SCM , meaning that M θ belongs to the TM-SCM class; (ii) G2:
M θ |= A ≤ , meaning that ≤ is one of the causal orders of M θ . (iii) G3: M θ |= A M , meaning that M θ satisfies Markovianity. (iv) G4: M θ |= A P V , meaning that M θ induces the observational distribution P V .
To satisfy G1, M θ must be a TM-SCM or one of its special subclasses. Furthermore, in this paper, we primarily focus on neural networks that provide the parameters θ for M θ , thereby referring to it as a neural TM-SCM. Previous works have constructed neural SCMs that satisfy this requirement in the scalar case; we categorize these works into four categories and propose corresponding prototype models as extensions to the vector case:
• DNME: Constraint the causal mechanism f i,θ to be a TM mapping, such that the causal mechanism
f i,θ (v pa(i) , u i ) = b i,θ (v pa(i) ) + a i,θ (v pa(i) ) ⊙ u i , (2
)
where a i,θ (v pa(i) ), b i,θ (v pa(i) ) ∈ R di with vector a i,θ always strictly positive, and ⊙ denotes component-wise multiplication. Since the Jacobian matrix of f i,θ w.r.t. u i is diagonal, it is named Diagonal Noise MEchanism, such as LSNM (Immer et al., 2023).
• TNME: Constraint the causal mechanism f i,θ to be a TM mapping, such that the causal mechanism
f i,θ (v pa(i) , u i ) = b i,θ (v pa(i) ) + A i,θ (v pa(i) ) u ⊺ i , (3
)
where
A i,θ (v pa(i) ) ∈ R di×di , b i,θ (v pa(i)
) ∈ R di with matrix A i,θ always a strictly positive lower triangular matrix. Since the Jacobian matrix of f i,θ w.r.t. u i is lower triangular, it is named Triangular Noise MEchanism, such as FiP (Scetbon et al., 2024).
• CMSM: Constraint the re-indexed solution mapping P ι • Γ θ to be a TM mapping by composing multiple TM mappings
P ι • Γ θ = T 1,θ • • • • • T n,θ ,(4)
where each T i,θ can be an autoregressive affine transformation in normalizing flows (Dinh et al., 2017). Since the solution map Γ θ is constructed by composing multiple mappings, it is named Composed Mapped Solution Mapping, such as CausalNF (Javaloy et al., 2023).
• TVSM: Constraints the re-indexed solution mapping P ι • Γ θ to be a TM mapping by defining it as the flow constructed from the solution to the ODE
dx(t) = v θ (x(t), t) dt, x(0) = v 0 ,(5)
where the vector field v θ is a Lipschitz continuous triangular mapping. According to the Picard-Lindelöf theorem, such flows are TMI, see Lemma A.21 in Appendix A.4. Since the solution map Γ θ is implied by a triangular velocity field, it is named Triangular Velocity Solution Mapping, such as CFM (Khoa Le et al., 2025).
The relationship between these related works and TM-SCM is detailed in Appendix B.4, and the implementation details of the four prototype models can be found in Appendix C.
To satisfy G2, the causal mechanisms f θ in the constructed SCM need to maintain a specific causal structure to derive the causal order ≤. When parameterizing the causal mechanisms f i,θ as in (Xia et al., 2023), we address the issue of inappropriate additional independence assumptions-arising from only knowing the causal order-by setting pa(i) = {j | j ≤ i, j ∈ I}. Similarly, when indirectly modeling the solution map Γ θ using an autoregressive generative model as in (Javaloy et al., 2023), we align the autoregressive order with the causal order.
To satisfy G3, the modeled exogenous distribution is required to satisfy P U θ = i∈I P U i,θ . To enhance the expressiveness of the exogenous distribution and to indirectly indicate that identifiability is independent of the specific implementation of the exogenous distribution, we employ unconstrained normalizing flows. Specifically, the log-likelihood of each P U i,θ is given by the change of variables formula:
log p U i,θ (u i ) = log p Z i,θ (T -1 i,θ (u i )) + log det J T -1 i,θ (u i ) ,(6)
where T i,θ : Z i → U i is MAF (Papamakarios et al., 2017),
and det J T -1 i,θ (u i ) is the Jacobian determinant at u i .
To satisfy G4, in generative model learning, the observational dataset D V is commonly used to optimize a learning objective, ensuring that the proxy SCM M θ induces an observational distribution P V,θ that closely approximates the true observational distribution P V . To achieve this, we adopt the traditional maximum likelihood (MLE) approach, corresponding to the negative log-likelihood (NLL) loss:
arg min θ - N i=1 log p V θ (v (i) ),(7)
where log p V θ is the log-likelihood of the modeled observational distribution. Since TM-SCM ensures a bijective Γ θ , the log-likelihood log p V θ (v (i) ) at v (i) can be computed using the change of variables formula.
this section cite: ['b28', 'b18', 'b41', 'b10', 'b20', 'b24', 'b54', 'b20', 'b30']

Section: Related Works
Isomorphism and counterfactual identifiability Several prior works have introduced notions similar to exogenous isomorphism and shown that SCMs satisfying these equivalence relations enjoy counterfactual consistency, mirroring the result of Theorem 3.2. This body of work includes counterfactual equivalence as defined in (Peters et al., 2017), which requires identical exogenous distributions; BGM equivalence, introduced for BGM within BSCM in (Nasr-Esfahany et al., 2023); LCM isomorphism, proposed by (Brehmer et al., 2022) for causal representation learning and proven, under additional conditions, to coincide with LCM counterfactual consistency; and domain counterfactual equivalence between ILDs, together with necessary and sufficient conditions, as defined in (Zhou et al., 2024). The latter three lines of work additionally assume that the solution mapping Γ is bijective, whereas Theorem 3.2 applies to arbitrary recursive SCMs.
TM-SCMs and their identifiability Previously, we classified four construction prototypes of TM-SCMs from a construction perspective; a complementary perspective considers the tasks these models address and the corresponding identifiability guarantees. For cause-effect identification, special TM-SCMs-including LiNGAM (Shimizu et al., 2006), ANM (Hoyer et al., 2008), PNL (Zhang & Hyvärinen, 2009), LSNM (Immer et al., 2023), and CAREFL (Khemakhem et al., 2021)-have been proven identifiable with respect to causal direction. For causal effect estimation, CausalNF (Javaloy et al., 2023) establishes representation identifiability of TMI, and similar TM-SCM constructions are adopted by StrAF (Chen et al., 2023), CCNF (Zhou et al., 2025), and CFM (Khoa Le et al., 2025), which therefore inherit the same theoretical guarantees. In counterfactual identification, (Lu et al., 2020, Theorem 1), (Nasr-Esfahany et al., 2023, Theorem 5.1), and (Scetbon et al., 2024, Theorem 2.14) each demonstrate that TM-SCM entails counterfactual identifiability under different settings, and these theorems are all special cases of Corollary 5.4.
Counterfactual inference with neural SCMs Proxy SCMs built with neural network components are widely employed for counterfactual reasoning by directly learning from observational or interventional data. Several methods focus on the tractability of inference rather than identifiability by adopting different neural modules, including DSCM (Pawlowski et al., 2020), Diff-SCM (Sanchez & Tsaftaris, 2022), and VACA (Sánchez-Martin et al., 2022). Other approaches obtain identifiability for a subset of counterfactual queries; for example, CVAE-SCM (Karimi et al., 2020) is identifiable for specific counterfactual queries from causal sufficiency and observational distribution, and NCM (Xia et al., 2023) shows a duality between the identifiability of structure-constrained proxy SCMs and non-parametric identification results. Once parametric assumptions are introduced, complete counterfactual identifiability becomes attainable, exemplified by any subclass of neural TM-SCMs. Additional examples are provided in Appendix B.4.
this section cite: ['b34', 'b29', 'b4', 'b59', 'b43', 'b16', 'b57', 'b18', 'b20', 'b6', 'b58', 'b24', 'b28', 'b41', 'b31', 'b38', 'b39', 'b21', 'b54']

Section: Experiments
To demonstrate that neural TM-SCM can effectively address the counterfactual consistency problem in practice, we conducted experiments on synthetic adatasets. These experiments were designed to showcase the model's ability to generate counterfactual results that are consistent with the test set, using only the endogenous samples drawn from the observational distribution as the training set. 3Datasets The experiments involve the following synthetic datasets, with details described in Appendix D.1.
• TM-SCM-SYM: A collection of four small datasets (BARBELL, STAIR, FORK, BACKDOOR) with up to 4 causal variables, using exogenous distributions that are standard or Markovian multivariate normals and manually defined TM causal mechanisms.
• ER-DIAG-50 and ER-TRIL-50: Each contains 50 datasets with 3-8 causal variables, Markovian multivariate normal exogenous distributions, and Erdős-Rényi causal graphs (edge probability 0.5). ER-DIAG-50 ensures diagonal Jacobians, while ER-TRIL-50 ensures lower triangular Jacobians for TM mappings.
this section cite: []

Section: Metrics
To evaluate trained M θ , we compute OBS WD (Wasserstein distance) for the fit to the observational distribution and CTF RMSE (root mean square error) for L 3consistency with ground truth counterfactual outcomes.  TM-SCM w/o O w/o M w/o T 0.5 1.0 1.5 2.0 2.5 0.0 0.2 0.4 0.6 0.8 0.2 0.3 0.4 0.5 0.6 0.0 0.2 0.4 0.6 0.8 0.30 0.35 0.40 0.45 0.50 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.25 2.30 2.35 2.40 2.45 2.50 2.55 2.60 0.0 0.1 0.2 0.3 0.4 0.5 0.6 Most w/o M settings result in higher CTF RMSE , emphasizing the role of A M . CMSM shows slightly lower stability than other methods. Additional results are in Appendix D.4.
this section cite: []

Section: Ablation on TM-SCM-SYM

this section cite: []

Section: Ablation on ER-DIAG-50 and ER-TRIL-50
We further conducted a more comprehensive evaluation on ER-DIAG-50 and ER-TRIL-50, reinforcing the generality of the theory through experiments on a wide variety of synthetic SCMs. The ablations in these experiments also targeted A ≤ , A M , and A TM-SCM .
Table 1 presents the final CTF RMSE on the ER-DIAG-50 and ER-TRIL-50 test sets for each method and its ablations.
The results demonstrate that violating A ≤ , A M , or A TM-SCM significantly degrades L 3 -consistency, emphasizing the importance of these assumptions and the validity of the theory to ensure consistent counterfactual inference. Additional results on these datasets can be found in Appendix D.4.
this section cite: []

Section: Conclusion
In this work, we explored ∼ L3 -identifiability, the strongest form of causal model indistinguishability within the PCH. By simplifying the problem through the introduction of exogenous isomorphism and ∼ EI -identifiability, we developed concrete methods to achieve identifiability in BSCMs and TM-SCMs. These findings unify and extend existing theories, providing reliability guarantees for counterfactual reasoning. Our empirical evaluations further validate the practicality of the proposed approach, paving the way for reliable applications in counterfactual modeling.
this section cite: []

Section: References
Ref_id:b0 Title: Identifiability of pathspecific effects Year: (2005)
Ref_id:b1 Title: On Pearl's Hierarchy and the Foundations of Causal Inference Year: ()
Ref_id:b2 Title: Representation learning: A review and new perspectives Year: (2013-08)
Ref_id:b3 Title: Foundations of structural causal models with cycles and latent variables Year: (2016)
Ref_id:b4 Title: Weakly supervised causal representation learning Year: (2022)
Ref_id:b5 Title: Modeling causal mechanisms with diffusion models for interventional and counterfactual queries Year: (2024)
Ref_id:b6 Title: Structured neural networks for density estimation and causal inference Year: (2023)
Ref_id:b7 Title:  Year: (2018)
Ref_id:b8 Title: Neural ordinary differential equations Year: (2018)
Ref_id:b9 Title: Nested counterfactual identification from arbitrary surrogate experiments Year: (2021)
Ref_id:b10 Title: Density estimation using real NVP Year: (2017)
Ref_id:b11 Title: Neural spline flows Year: (2019)
Ref_id:b12 Title: Interpolating between optimal transport and mmd using sinkhorn divergences Year: (2019)
Ref_id:b13 Title: Review of causal discovery methods based on graphical models Year: (2019)
Ref_id:b14 Title: Scalable reversible generative models with free-form continuous dynamics Year: (2019)
Ref_id:b15 Title: Independent mechanism analysis, a new concept Year: (2021)
Ref_id:b16 Title: Nonlinear causal discovery with additive noise models Year: (2008)
Ref_id:b17 Title: Nonlinear independent component analysis for principled disentanglement in unsupervised deep learning Year: (2023)
Ref_id:b18 Title: On the identifiability and estimation of causal location-scale noise models Year: (2023-07)
Ref_id:b19 Title: Sum-of-squares polynomial flow Year: (2019-06)
Ref_id:b20 Title: Causal normalizing flows: from theory to practice Year: (2023)
Ref_id:b21 Title: Algorithmic recourse under imperfect causal knowledge: a probabilistic approach Year: (2020)
Ref_id:b22 Title: Variational autoencoders and nonlinear ica: A unifying framework Year: (2020-08)
Ref_id:b23 Title: Causal autoregressive flows Year: (2021-04)
Ref_id:b24 Title: Learning structural causal models from ordering: Identifiable flow models Year: (2025-04)
Ref_id:b25 Title: Identifiability of deep generative models without auxiliary information Year: (2022)
Ref_id:b26 Title: Counterfactual fairness Year: (2017)
Ref_id:b27 Title: Transport-based counterfactual models Year: (2024)
Ref_id:b28 Title: Sample-efficient reinforcement learning via counterfactual-based data augmentation Year: (2020)
Ref_id:b29 Title: Counterfactual identifiability of bijective causal models Year: (2023-07)
Ref_id:b30 Title: Masked autoregressive flow for density estimation Year: (2017)
Ref_id:b31 Title: Deep structural causal models for tractable counterfactual inference Year: (2020)
Ref_id:b32 Title: Causality: Models, Reasoning and Inference Year: (2009)
Ref_id:b33 Title: The Book of Why: The New Science of Cause and Effect Year: (2018)
Ref_id:b34 Title: Elements of Causal Inference: Foundations and Learning Algorithms Year: (2017)
Ref_id:b35 Title: Causal Models for Dynamical Systems Year: ()
Ref_id:b36 Title: Counterfactual harm Year: (2022)
Ref_id:b37 Title: Normalizing Flows in PyTorch Year: (2024)
Ref_id:b38 Title: Diffusion causal models for counterfactual estimation Year: (2022-04)
Ref_id:b39 Title: Designing variational graph autoencoders for causal queries Year: (2022)
Ref_id:b40 Title: One-dimensional issues Year: (2015)
Ref_id:b41 Title: A fixed-point approach for causal generative modeling Year: (2024-07)
Ref_id:b42 Title: Toward causal representation learning Year: (2021)
Ref_id:b43 Title: A linear non-gaussian acyclic model for causal discovery Year: (2006)
Ref_id:b44 Title: Complete identification methods for the causal hierarchy Year: (2008)
Ref_id:b45 Title: Causal discovery and inference: concepts and recent methodological advances Year: (2016-02)
Ref_id:b46 Title: Probabilities of causation: Bounds and identification Year: (2000)
Ref_id:b47 Title: Finding counterfactually optimal action sequences in continuous state spaces Year: (2023)
Ref_id:b48 Title: Nonparametric identifiability of causal representations from unknown interventions Year: (2023)
Ref_id:b49 Title: D'ya like dags? a survey on structure learning and causal discovery Year: (2022-11)
Ref_id:b50 Title: Unconstrained monotonic neural networks Year: (2019)
Ref_id:b51 Title: Learning counterfactual outcomes under rank preservation Year: (2025)
Ref_id:b52 Title: Indeterminacy in generative models: Characterization and strong identifiability Year: (2023-04)
Ref_id:b53 Title: The causal-neural connection: Expressiveness, learnability, and inference Year: (2021)
Ref_id:b54 Title: Neural causal models for counterfactual identification and estimation Year: (2023)
Ref_id:b55 Title: Relating graph neural networks to structural causal models Year: (2021)
Ref_id:b56 Title: Fairness in decision-making -the causal explanation formula Year: (2018-04)
Ref_id:b57 Title: On the identifiability of the post-nonlinear causal model Year: (2009)
Ref_id:b58 Title: Causally consistent normalizing flow Year: (2025-04)
Ref_id:b59 Title: Towards characterizing domain counterfactuals for invertible latent causal models Year: (2024)
