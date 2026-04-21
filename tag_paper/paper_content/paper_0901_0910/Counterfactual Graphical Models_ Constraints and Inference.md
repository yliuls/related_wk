Title: Counterfactual Graphical Models: Constraints and Inference
Abstract: Graphical models have been widely used as parsimonious encoders of the constraints underlying probability models. When organized in a structured way, these models can facilitate the derivation of non-trivial constraints, the inference of quantities of interest, and the optimization of their estimands. In particular, causal diagrams enable the efficient representation of the structural constraints of the underlying causal system. In this paper, we introduce an efficient graphical construction called Ancestral Multi-world Networks that is sound and complete for reading counterfactual independences from a causal diagram using d-separation. Moreover, we introduce the counterfactual (ctf-) calculus, which can be used to transform counterfactual quantities using three rules licensed by the constraints encoded in the diagram. This result generalizes Pearl's celebrated do-calculus from interventional to counterfactual reasoning.

Section: Introduction
Counterfactuals form the basis of important notions across human cognition that require retrospective thinking, where one must compare what did happen in the real world versus what would have happened under some different hypothetical conditions. Given the impossibility of observing an alternative outcome once an action is taken, counterfactuals evoke "what if?" questions whose answers can only be approached by imagining hypothetical conditions contrary to this factual evidence. For instance, questions such as "what would be the death rates had the vaccination started two weeks earlier?" or "given that I arrived late, would I have been on time had I taken the subway instead of the taxi?" require us to carry out a mental experiment where Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). we recover some state of affairs, perform a change in the sequence of events, and let a hypothetical situation to play out. More generally, counterfactuals are an important component in the construction of explanations regarding why certain events occurred the way they did (Pearl, 2000;Pearl & Mackenzie, 2018;Bareinboim et al., 2020).
One fundamental topic of study in counterfactual reasoning is understanding the various quantities, the constraints on their relation, and the types of inferences allowed across various counterfactual worlds. Specifically, counterfactual quantities evoke hypothetical conditions that could contradict the factual evidence, underpinning different applications involving blame and responsibility, credit assignment, and more individualized types of decisions (Pearl, 2000). Examples of such quantities include the effect of treatment on the treated (Heckman, 1992;Pearl, 2000), path-specific effects (Pearl, 2001;Avin et al., 2005), and causal and spurious variations (Zhang & Bareinboim, 2018;Plečko & Bareinboim, 2024). There are also quantities such as the probability of necessity (PN), probability of sufficiency (PS), and the probability of necessity and sufficiency (PNS) that relate to fundamental aspects of how events are related and can explain the other. For example, consider the causal diagram in Figure 1 over the variables age, treatment, and survival. The counterfactual event (Y x = 1 | X = x ′ ) refers to the survival of a person (Y = 1) that gets a treatment X = x when they would naturally decide not to get treated (x ′ ). Such queries depict a quintessential counterfactual situation, since we aim to evaluate a world that contradicts the factual one in which the person was not treated.
In the first part of our paper, we revisit and generalize counterfactual constraints -exclusion and independence restrictions, and consistency (Pearl, 2000) -and show how they follow from the Structural Causal Model semantics. Specifically, we introduce a new graphical representation that encodes independences between counterfactual random variables, which we call Ancestral Multi-World Network (AMWN). Based on this new data structure, we formally show that the d-separation criterion is complete for reading such constraints based on a causal graph and a set of counterfactual variables. Compared with the prior literature, the newly proposed method improves over the Twin Networks (Balke & Pearl, 1994), for which d-separation is not complete, and from Single World Intervention Graphs (Richardson & Robins, 2013), which consider a single intervention at a time. AMWN also differs from Multi-Networks and Counterfactual Graphs (Shpitser & Pearl, 2007), which were conjectured to be complete but require constructing a possibly exponential number of graphs to test separation among counterfactual variables rather than events.
In the second part of the paper, and building on the constraints and AMWN construction, we formulate a set of three rules for counterfactual inference called Counterfactual Calculus (ctf-calculus). Compared with the literature, our rules are more general than Pearl's celebrated do-calculus (Pearl, 1994;1995) for interventional reasoning, since it allows for the transformation of counterfactual quantities to infer the implied equality constraints. Moreover, we show that the counterfactual calculus is complete for identifying counterfactuals from observational and interventional distributions. This set of rules also differs from the Potential Outcome Calculus (po-calculus) (Malinsky et al., 2019), which has been shown to hold if and only if the corresponding do-calculus rules hold. While po-calculus rules require counterfactual variables to follow certain patterns in terms of interventions and require pre-processing steps to be used for certain identification tasks, we propose rules supporting more general mixes of interventions, which, combined with probability axioms, are sufficient for deciding counterfactual identification.
More specifically, our contributions are as follows:
1. Graphical criteria: a sound, complete, and efficient procedure to test conditional independences among counterfactual variables using d-separation on a modified causal diagram.
2. Inference rules: a set of inference rules for counterfactual reasoning that are sound and complete for counterfactual identification from observational and experimental distributions.
Proofs can be found in the supplemental material.
Definitions and Background. We denote variables by capital letters, X, and values by small letters, x. Bold letters, X represent a set of variables and x a set of values. The domain of a variable X is denoted by Val(X). Two values
x and z are consistent if they share the common values for X ∩ Z. We also denote by x \ Z the value of X \ Z consistent with x and by x∩Z the subset of x corresponding to variables in Z. We assume the domain of every variable is finite.
We represent qualitative assumptions using causal graphs, denoted with a calligraphic letter, e.g., G, etc. Given a graph G, G WX is the result of removing edges coming into variables in W and going out from variables in X. G[W] denotes a vertex-induced subgraph, which includes W and the edges among its elements. We use kinship notation for graphical relationships such as parents, children, descendants, and ancestors of a set of variables.
We base our analysis on the Structural Causal Model (SCM) paradigm (Pearl, 2000). An SCM M is a 4-tuple ⟨U, V, F, P (u)⟩, where U is a set of exogenous (latent) variables; V is a set of endogenous (observable) variables; F is a collection of functions such that each variable V i ∈ V is determined by a function f i ∈ F. Each f i is a mapping from a set of exogenous variables U i ⊆ U and a set of endogenous variables Pa
i ⊆ V \ {V i } to the domain of V i .
Uncertainty is encoded through a probability distribution over the exogenous variables, P (U).
An SCM M induces a causal diagram G where V is the set of vertices, there is a directed edge (V j → V i ) for every V i ∈ V and V j ∈ Pa i , and a bidirected edge (Bareinboim et al., 2020). We assume that the underlying model is recursive. That is, there are no cyclic dependencies among the variables. SCMs allow us to define counterfactual quantities with precision based on the Pearl's Causal Hierarchy (PCH) (Pearl & Mackenzie, 2018;Bareinboim et al., 2020). This hierarchy is divided into three layers (Figure 2): the first one (L 1 ) captures the notion of "seeing," that is, observing a certain phenomenon or reality and possibly making inferences about it. The second (L 2 ) allows one to represent the notion of "doing", that is, intervening (or deliberately acting) in the environment to bring about a certain state of affairs. Modifying an SCM gives natural valuations for quantities of this kind, as defined next. Definition 1.1 (Submodel). Let M be a causal model, X a set of variables in V, and x a particular realization of X. A submodel M x of M is the causal model
(V i V j ) for every pair V i , V j ∈ V such that U i ∩ U j ̸ = ∅ (V i and V j have a common exogenous parent)
M x = ⟨U, V, F x , P (U)⟩, where(1)
F x = {f i : V i / ∈ X} ∪ {X ← x}.(2)
That is, performing an external intervention (or action) is modeled through the replacement of the original (natural) mechanisms associated with some variables X with a constant x, which is represented by the do-operator. The impact SCM L 1 : P (V) of the intervention on an outcome variable Y is commonly called the potential response:
L 2 : P (V x ) L 3 : P (V 1[x1] , V 2[x2] , . . .)
Definition 1.2 (Potential Response). Let X and Y be two sets of variables in V, and u be a unit. The potential response Y x (u) is defined as the solution for Y of the set of equations F x with respect to SCM M (for short,
Y Mx (u)). That is, Y x (u) = Y Mx (u).
In other words, Y Mx (u) is obtained through the computation of Y(u) in the submodel M x . On the other hand, the meaning of every term in the counterfactual layer (L 3 ) can be directly determined from a fully specified structural causal model, as described in the sequel:
Definition 1.3 (Counterfactual Distribution Valuation). An SCM M = ⟨U, V, F, P (U)⟩ induces a family of joint distributions over counterfactual events Y x , . . . , Z w , for any Y, Z, . . . , X, W ⊆ V, P M (y x , . . . , z w ) is given by: u 1[Y x (u) = y, . . . , Z w (u) = z ] P (u).(3)
Let W * = {(W 1 ) T1 , (W 2 ) T2 , . . .} represent an arbitrary set of counterfactual variables such that W i ∈ V and T i ⊆ V for i = 1, . . . , l. We assume throughout this paper that all the distributions generated by the models are positive.
this section cite: ['b9', 'b11', 'b9', 'b5', 'b9', 'b10', 'b0', 'b12', 'b9', 'b1', 'b13', 'b14', 'b7', 'b8', 'b6', 'b9', 'b11']

Section: Counterfactual Constraints
We begin by stating three types of constraints that hold over counterfactuals random variables: consistency (Section 2.1), exclusion (Section 2.2), and independence (Section 2.3), which we detail in the following subsections.
this section cite: []

Section: Consistency Constraints
Consistency constraints relate to the interplay between observing a variable taking a particular value and the effect of an intervention that fixes this variable to the same value.
To ground this idea, consider an SCM M over endogenous variables V = {X, Y, Z} and suppose we are interested in studying the joint counterfactual event (Y x = y, X = x).
Following the proper semantics (Theorem 1.2), the value of variable X is given by the solution of the system of equations F associated with M, X(u), for each unit U = u.
Similarly, the value of Y x is given by the solution of the system F x , Y x (u), for the same unit. The event X = x occurs for u whenever the solution of f x is equal to x. While f x is fixed as a constant x in F x (as illustrated in Figure 3(a)), for any unit U = u for which X = x, the result of these two systems of equations coincide.
Both models will match in the value of every observable, i.e., for u
′ = {u | X(u) = x}, X(u ′ )=X x (u ′ )=x, Y (u ′ )=Y x (u ′ ), Z(u ′ )=Z x (u ′ ). (4)
Moreover, the probability of the corresponding random variables follows from averaging P (U) for those u, and then:
P (Y x = y, X = x) = u 1[Y x (u) = y, X(u) = x] P (u) (5) = u 1[Y (u) = y, X(u) = x] P (u) (6) = P (Y = y, X = x). (7) Again, this is so because Y x (u) = Y (u) for those u for which X(u) = x.
More generally, when considering all the endogenous variables, we have:
P (Y x , Z x , X = x) = P (Y, Z, X = x),(8)
In other words, once we restrict our attention to the set of units that generate X = x, then the variations of Y x , and Z x are consistent with the variations of Y , and Z, respectively.
Intuitively, once X takes the value x, naturally, other variables in the model behave the same as if X had been fixed to x by intervention, for instance, Y x = Y . 1
More broadly, consistency does not depend on the independence structure among the exogenous variables, P (U), and follows from the relationships within the structural mechanisms F.
The following characterizes this family of constraints across endogenous variables:
Lemma 2.1 (Consistency).
Given SCM M and X, Y ∈ V, T * be any combination of counterfactuals, and let x be a value in the domain of X. Then,
P (Y T * , X T * = x) = P (Y T * x , X T * = x).(9)
As suggested by the term T * in Theorem 2.1, consistency between observations and interventions not only occurs for interventions that fix a variable to a constant value (e.g., do (X = x)) but is also true with interventions that set a variable to match another counterfactual variable, as discussed next.
1 One way to interpret such a statement is through the independence of the mechanisms that give value to each of the endogenous variables in the system in conjunction with the locality of the intervention.
(u | X = x) F F x f z f x f y f z f ′ x f y (X = x, Y x ) (a) Mechanisms involved in generating the event (Yx, X = x).
(u)
F Xz F z f z f ′ x f y f ′ z f x f y (Y Xz ) (b)
The variable YX z results from forcing X to take the value Xz(u) for every u. 2.1.1. NESTED COUNTERFACTUALS So far we have considered counterfactuals of the form Y x , where the subscript x indicates that an intervention do (X = x) has been performed in the system. We turn our attention to interventions that could be expressed as do (X = X z ), and represent settings where the variable X is set to behave as another counterfactual variable, say X z . This operation is illustrated in Figure 3(b). In other words, the value of X z is computed in a sub-model F z (figure's r.h.s.) and used to replace the natural mechanism f x (in the l.h.s.). A random variable Y in such a system is represented with a counterfactual of the form Y Xz , which is called a nested counterfactual.
The nesting means the target counterfactual internally refers to another nested world (possibly multiple times). Corollary 2.2 (Counterfactual Unnesting (CU)). Let Y, X ∈ V, T, Z ⊆ V, and let z be a set of values for Z. Then, the nested counterfactual P (Y T * Xz = y) can be written with one less level of nesting as:
P (Y T * Xz = y) = x P (Y T * x = y, X z = x). (10
)
This statement follows from the law of total probability and consistency itself, i.e.:
P (Y T * Xz = y) = x P (Y T * X z =y, X z =x) (sum over X z ) (11) = x P (Y T * x =y, X z = x ) (consistency). (12
)
These two steps allow us to reason about nested counterfactuals and transform them into expressions involving nonnested ones.
this section cite: []

Section: Exclusion Constraints
Although the semantics of counterfactuals allows one to consider a variable Y t for arbitrary Y ∈ V and T ⊆ V, some counterfactual variables are not entirely free to vary depending on the topology and the sparsity of the causal system. For example, consider the simple chain graph in Figure 4(a) and the counterfactual variables Y z and Y zx . To understand the relationship between these two variables, we write the corresponding sub-models M z and M zx :
Fz=      Xz ← fX (Ux) Zz ← z Yz ← fY (z, Uy) Fzx=      Xzx ← x Zzx ← z Yzx ← fY (z, Uy),(13)
and P (U) = P (U x )P (U z )P (U y ). Note that for each unit U = u, the variables Y z and Y zx are the same. Intuitively, once the value of Z is fixed to z by intervention, the only source of variation for the variable Y in both M z and M zx comes from U y , so intervening on X is irrelevant. In some sense, the intervention on X can be excluded without any changes in the value of Y , which gives the name exclusion restriction.
In graphical terms, an intervention on a variable X could affect another variable Y only if there exists a causal (directed) path from X to Y in G. 2 Although in Figure 4(a) there is such path, the same is severed once Z is intervened on. This observation can be stated more generally in the form of an operator used to exclude interventions from a given counterfactual variable as follows:
Lemma 2.3 (Exclusion operator). Let Y x be a counterfactual variable, G a causal diagram, and
Y z such that Z = X ∩ An(Y ) G X and z = x ∩ Z. (14
)
Then, Y z = Y x holds for any model compatible with G.
Moreover, this transformation is denoted as ∥Y x ∥:= Y z .
Note that by keeping X ∩ An(Y ) G X (Equation ( 14)), the exclusion operator removes from the counterfactual's antecedent (i.e., subscript) variables that are not ancestors of Y (variables without causal paths to Y ) as well as those ancestors that once do (X) is performed are no longer ancestors of Y .
For a set Y * , define ∥Y * ∥= Yt∈Y * ∥Y t ∥. The result of applying the exclusion operator to Y x , ∥Y x ∥, is always equal to Y x or an equivalent counterfactual variable with fewer variables in its antecedent.
Z X Y (a) Chain causal structure. Z W X Y (b) Causal diagram over 4 variables. One interesting feature of exclusion constraints is that they are derivable from the order relative to the mechanisms, F, of the underlying SCM, M * . Other invariances from M * come from sparsity in P (U), as discussed in the sequel.
this section cite: []

Section: Independence Constraints and the Counterfactual d-separation Criterion
The ability to represent multiple worlds simultaneously is a fundamental aspect that sets apart the third layer of the PCH from the others. One could therefore consider a probability expression involving variables from multiple worlds, such as Y x and Z x ′ when x ̸ = x ′ .
At the structural level, multiple interventions entail different copies of the mechanisms F of the SCM, each for a different world (syntactically represented by a different subscript), but all sharing the same P (U). As implied by Equation ( 3), a counterfactual distribution can be evaluated by passing the set of exogenous variables U through the different versions of those mechanisms, depending on which hypothetical world one aims to evaluate. This process can be mimicked and represented at the graphical level by a "meta" diagram incorporating different instances of the endogenous variables produced by the various mechanisms and connecting different worlds through the U variables. This idea allows the evaluation of separation statements among nodes representing counterfactual variables, which in turn imply conditional independences among the corresponding variables in the underlying distribution.
For concreteness, consider whether the causal graph in Figure 4
(b) implies that (Y xw , W x ′ ⊥ ⊥ X | Z x ′ ). Figure 5(a)
shows a natural generalization of the twin network to 3 worlds, a 3-plet network, for this graph and question. Note that the variables in the query involve three submodels: M, M x ′ , and M xw , all depicted in the network sharing explicit unobservable variables.
While it seems that X is d-connected to Y xw given Z x ′ in Figure 5(a), due to the active path X ← Z ← U z → Z xw → Y xw , the exclusion operator reveals Z x ′ = ∥Z x ′ ∥= Z. This means that conditioning on Z x ′ is the same as conditioning on Z, and the separation holds.
In this sense, we should merge the nodes Z, Z x ′ , and Z xw due to the deterministic relationship among them. It is also convenient to ignore nodes of variables fixed by intervention and reduce every variable with the exclusion operator. This results in the 3-plet network shown in Figure 5(b). In this new graph, d-separation can be used to tell that X and Y xw = Y w are separated given Z x ′ = Z.
More generally, we can construct twin networks, 3-plet networks, or k-plet networks depending on the number of interventions in the separation statement. Then, use the exclusion operator to merge nodes corresponding to variables that are deterministically the same. This method, however, includes many variables in the graph that we do not need to check. To improve efficiency, as we will show later, we discuss the concept of ancestors of a counterfactual.
Definition 2.4 (Ancestors (of a counterfactual) (Correa et al., 2021)). Let Y x be such that Y ∈ V, X ⊆ V. Then, the set of (counterfactual) ancestors of
Y x , denoted by An(Y x ), consist of each W z such that W ∈ An(Y ) G X \ X (which includes Y itself), and z = x ∩ An(W ) G X .
This extends the idea that, in graphical terms, a variable can only affect another if the former is an ancestor of the latter.
When counterfactuals are involved, some of those ancestors become irrelevant (by virtue of the exclusion operator). For example, for the graph in Figure 4
(a), X is an ancestor of Y , but it is not a (counterfactual) ancestor of Y z , because under do (Z), X cannot affect Y z . Similarly, Z is an ancestor of Y , but for Y x a counterfactual ancestor is Z x , not X or Z.
For a set of variables W * , we define An(W * ) as the union of the ancestors of each variable in the set. That is, An(W * ) = Wt∈W * An(W t ). For example, in Figure 4
(b), An(Y xw ) = {Y w , Z}, An(W x ′ ) = {W x ′ }, An(X) = {X, Z}.
We describe a graphical construction called the Ancestral Multi-World Network (AMWN), denoted G A (G, Y * ). This data structure is a function of the original causal diagram G and the counterfactual variables Y * in the separation statement to be evaluated. Algorithm 1 describes the procedure for creating an AMWN.
For concreteness, let us consider again the evaluation of the separation query (Y xw , W x ′ ⊥ ⊥ X) using the causal diagram in Figure 4(b). In line 1, the procedure computes An(Y * ), which will be added as nodes in the AMWN.
The associated directed arrows witness the ancestrality of the variables involved. For instance, Z is an ancestor (parent) of Y w , hence Z and the arrow Z → Y w must be present in the graph. Note that, at this point, the resulting graph is a subgraph of Figure 5(b), but the rest of the graph is not relevant to evaluate the separation statement with d-separation. That is,
(X t ⊥ ⊥ Y r | Z * ) can be judged using d-separation on top of G A (X t , Y r , Z * ).
The second part of Algorithm 1 explicitly adds latent vari-  ables U to facilitate reasoning about the relations among variables appearing more than once in the graph with different subscripts and variables originally connected by latent confounding. 3   For instance, consider the causal diagram in Figure 6(a) and whether (Y xw ⊥ ⊥X | {Z, W }). The relevant set of ancestors is An(Y xw , X, Z, W ) = {Y xw , Z w , X, Z, W }. The corresponding AWMN is shown in Figure 6(b). The node U z has been added and connected to Z and Z w (line 1), which come from the same original variable. There is also the node U zx , that is connected to Z, Z w and X due to the bidirected arrow Z X in G (line 1). By the d-separation criterion, the path X Z w → Y xw is active given {Z, W }, which leads to the conclusion that (Y xw ⊥ ̸ ⊥ X | Z, W ), as stated in the sequel.
M Mx′ Mxw Z W X Y Zxw Wxw Xxw Yxw Zx′ Wx′ Xx′ Yx′ Ux Uz Uw Uy (a) M Mx′ Mw Z W X Y Yw Wx′ Yx′ Ux Uz Uw Uy (b) Z W x ′ X Y w (c)
W Z Y X (a) W Z Y xw X Z w U z U zx (b)
this section cite: ['b4']

Section: Theorem 2.5 (Independence Constraints -Counterfactual d-separation (soundness)). Consider a causal diagram G
3 The exogenous variables shared across worlds are precisely the anchors of invariance in these settings. They represent precisely the identity of the units submitted to these different counterfactual conditions.
this section cite: []

Section: Algorithm 1 AMWN-CONSTRUCT(G, Y * )
Input: Causal diagram G and a set of counterfactual variables Y * . Output: GA(Y * ) the AMWN of G and Y * .
1: Initialize a network G ′ adding the variables in An(Y * ) together with the directed arrows witnessing the ancestrality. 2: for each variable V ∈ V appearing more than once in G ′ do 3: Add a node UV and an edge UV → Vx for every instance of Vx of V . 4: end for 5: for each bidirected V W where V and W are in G ′ do 6: Add a node UV W and edges from it to Vx and Wx, for every instance of
Vx of V or Wx of W in G ′ . 7: end for 8: return G ′ .
and a collection of counterfactual distributions, P ⋆⋆⋆ , induced by the SCM associated with G. For counterfactual variables X t , Y r , Z * ,
(∥X t ∥⊥ ⊥∥Y r ∥ ∥Z * ∥) G A → (∥X t ∥⊥ ⊥∥Y r ∥ ∥Z * ∥) P ⋆⋆⋆ .
Moreover, if an independence relationship cannot be inferred with the criterion, there exist at least two models inducing the graphical model where the independence does not hold.
In other words, the d-separation criterion is sound for the counterfactual variables in the AMWN, that is, if ∥X t ∥ and ∥Y r ∥ are d-separated given ∥Z * ∥ in the diagram G A (X t , Y r , Z * ), then X t and Y r are independent given Z * in every distribution P ⋆⋆⋆ compatible with the causal diagram G. It is also complete because if d-separation does not hold in the AMWN, then independence cannot be guaranteed.
This result allows us to use the construction of an AMWN of G to test whether a pair of counterfactual variables is independent in the probability distributions generated by the model compatible with G. Now, we examine the time complexity of constructing an AMWN. Let z be the number of different interventions in the separation query, and n, m be the number of nodes and edges, respectively. In line 1, the set of counterfactual ancestors can be computed in time linear to the size of the graph, for each intervention appearing in Y * ; hence the step takes time O(z(n + m)). Due to line 1, no more than n latent nodes and zn edges are added. Line 1 adds m latent nodes and 2zm edges at most. Overall, the construction takes O(z(n + m)), which is polynomial in the size of G and Y * .
The resulting graph G A has O(z(n + m)) nodes and edges, hence running d-separation on top of it takes O(z(n + m)) time (van der Zander et al., 2014). Compared with the classical d-separation criterion, the time required to use AMWN increases by a factor of z, the number of different worlds involved in the query. Table 1 summarizes the methods discussed, in terms of whether they allow for checking any separation constraints (among any counterfactual in the considered worlds), if d-separation is complete for them, and the time complexity of the construction of the graph and checking the constraint. 4 Overall, the method based on AMWN is more general and efficient than previous algorithms in the literature.
this section cite: []

Section: The Counterfactual Calculus
Building on our understanding of the constraints discussed earlier, this section introduces the counterfactual calculus and how it can be used for counterfactual inference based on the assumptions encoded in a causal diagram.
In the spirit of Pearl's celebrated interventional calculus (do-calculus), its counterfactual counterpart allows one to transform expressions in the form P
Rule 2 (Independence Rule -Adding/removing counterfactual observations)
P (y r | x t , w * ) = P (y r | w * ) if (Y r ⊥ ⊥ X t | W * ) in G A ,(16)
Rule 3 (Exclusion Rule -Adding/removing interventions)
P (y xz , w * ) = P (y z , w * ) if X ∩ An(Y) = ∅ in G Z , (17
)
where G A is the counterfactual ancestral graph G A (G, Y r ∪ X t ∪ W * ).
The first rule of the calculus, consistency, was discussed in Section 2.1. One distinct feature of this rule is that it does not depend on the graphical structure and allows for adding or removing interventions whenever a specific observational context and the antecedent of the counterfactual (subscript) match. As mentioned earlier, consistency is essentially the probabilistic instantiation of the invariances that follow from the modularity and stability of the causal mechanisms of the underlying system.
The second rule, independence (Section 2.3), corresponds to a generalized version of d-separation for counterfactual events. Syntactically, it permits the addition/removal of counterfactual evidence in a probability distribution.
The third rule, exclusion (Section 2.2), follows from the idea that interventions on variables without a causal path to the observed variable do not affect this variable and, therefore, can be dismissed. 6For concreteness, we illustrate next the use of the ctfcalculus rules for counterfactual identification tasks through a few examples.
Example 1 (ETT in the Backdoor diagram). Consider the causal diagram in Figure 1 and the observational distribution as input, and the counterfactual distribution P (y x | x ′ ) as the query. Using the ctf-calculus, we can then write: (R1:
P (y x | x ′ ) = z P (y x | z, x ′ )P (z | x ′ ) (Conditioning on Z)(18)
= z P (y x | z x , x ′ )P (z | x ′ ) (R3: {X} ∩ An(Z) = ∅) (19
) = z P (y xz | z x , x ′ )P (z | x ′ ) (R1: (Z x = z ⇒ Y x = Y xz )) (20
) = z P (y xz | z, x ′ )P (z | x ′ ) (R3: {X} ∩ An(Z) = ∅) (21
) = z P (y xz | z, x)P (z | x ′ ) (R2: (X ⊥ ⊥ Y xz | Z) in G A (Figure 7(a))) (22
) = z P (y | z, x)P (z | x ′ ) X Z Y xz(
(Z = z, X = x ⇒ Y xz = Y )) (23
)
The effect of the treatment on the treated is then identifiable from P (x, z, y) and G. ■ Example 2 (Natural Direct and Indirect Effects). Consider the causal diagram in Figure 7(b) and suppose we wish to evaluate the natural direct (NDE) to understand how exercising (X) affects cardiovascular disease (Y ) by means other than affecting the cholesterol level (W ), that acts as a mediator of this relationship. The NDE can be written in counterfactual language as
N DE x,x ′ (y) = P (y x ′ ,Wx ) -P (y x ).(24)
The derivation of the first term of the NDE expression goes as follows:
P (y x ′ Wx ) = w P (y x ′ w , w x ) (CU, cor. 2.2: sum over W x + consistency) (25
) = w P (y x ′ w | w x )P (w x ) (Chain rule) (26
) = w P (y x ′ w | w x ′ )P (w x ) (R2: (Y x ′ w ⊥ ⊥ W x , W x ′ ) in G A (Figure 8(a))) (27) = w P (y x ′ w | w x ′ , x ′ )P (w x ) (R2: (Y x ′ w ⊥ ⊥ X | W x ′ ) in G A (Figure 8(b))) (28) = w P (y x ′ w | w, x ′ )P (w x ) (R1: (X = x ′ ⇒ W x ′ = W )) (29
) = w P (y | w, x ′ )P (w x ) (R1: (W = w, X = x ′ ⇒ Y x ′ w = Y )).(30)
Here, P (w x ) cannot be further reduced to an expression in terms of observational distributions.
For the baseline P (y x ), the derivation goes as follows:
P (y x ) = w P (y x | w x )P (w x ) W x W x ′ Y x ′ w (a) X W x ′ Y x ′ w (b) X W x Y x (c) Figure 8. Causal diagrams used in derivation in Example 2. (Condition on W x ) (31
)
= w P (y x | w x , x)P (w x ) (R2: (Y x ⊥ ⊥ X | W x ) in G A (Figure 8(c))) (32
)
= w P (y | w, x)P (w x ) (R1: (X = x ⇒ W x = W, Y x = Y )).(33)
Finally, we get
NDE x,x ′ (y)= w (P (y|w, x ′ )-P (y|w, x)) P (w x ). (34
)
by putting Equations ( 30) and ( 33) together. ■
The calculus guarantees the correctness of the reduction whenever such a derivation from a counterfactual query to the probabilities over the observed distributions is available.
Theorem 3.2 (Soundness and Completeness for Counterfactual Identifiability). A counterfactual quantity Q = P (y * | x * ) is identifiable from a given combination of observational and experimental distributions and a causal diagram G if and only if there exists a sequence of applications of the rules of ctf-calculus and the probability axioms that reduces Q into a function of the available distributions.
In other words, any derivation following the ctf-calculus is correct (soundness) and, if any counterfactual is identifiable from certain observational (L 1 ) and interventional (L 2 ) distributions, there must exist a sequence of applications of the ctf-calculus that witnesses the mapping of the available distributions and the target effect (completeness).
Since any causal effect can be written in counterfactual terms, it is only natural that ctf-calculus subsumes docalculus, as follows.
this section cite: []

Section: Lemma 3.3 (ctf-calculus -do-calculus reduction). ctfcalculus subsumes do-calculus.
More details on the relationship between do-calculus and ctf-calculus can be found in Appendix C.2 in (Correa & Bareinboim, 2024).
this section cite: ['b3']

Section: Conclusions
In this paper, we first established consistency (Theorem 2.1), exclusion (Theorem 2.3), and independence constraints (Theorem 2.5) following from the SCM semantics. We showed that d-separation is complete for obtaining independence constraints from the causal diagram using an efficient graphical construction called Ancestral Multi-World Network (Algorithm 1). This constitutes the first efficient procedure for reading counterfactual independence. We then introduced a set of rules called counterfactual calculus (Theorem 3.1), which can be used to transform target counterfactual quantities based on the constraints encoded in the diagram. Finally, we showed that counterfactual calculus is sound and complete for identifying counterfactuals from an arbitrary combination of observational and experimental distributions (Theorem 3.2). We hope the results in this paper can further our understanding and expand the toolbox for performing causal reasoning, closing a journey that started with Pearl's fundamental results on d-separation for observational distributions (circa 1986) and the do-calculus for interventional reasoning (1995). We now have more general machinery that allows for reasoning across the three layers of the causal hierarchy, including the very top: counterfactual relations.
this section cite: []

Section: References
Ref_id:b0 Title: Identifiability of Path-Specific Effects Year: (2005)
Ref_id:b1 Title: Probabilistic evaluation of counterfactual queries Year: (1994)
Ref_id:b2 Title: On Pearl's Hierarchy and the Foundations of Causal Infer-ence Year: ()
Ref_id:b3 Title: Counterfactual graphical models: Constraints and inference Year: (2024-08)
Ref_id:b4 Title: Nested Counterfactual Identification from Arbitrary Surrogate Experiments Year: (2021)
Ref_id:b5 Title: Randomization and Social Policy Evaluation Year: (1992)
Ref_id:b6 Title: A potential outcomes calculus for identifying conditional pathspecific effects Year: (2019)
Ref_id:b7 Title: A probabilistic calculus of actions Year: (1994)
Ref_id:b8 Title: Causal diagrams for empirical research Year: (1995)
Ref_id:b9 Title: Causality: Models, Reasoning, and Inference Year: (2000)
Ref_id:b10 Title: Direct and indirect effects Year: (2001)
Ref_id:b11 Title: The Book of Why Year: (2018)
Ref_id:b12 Title: Causal fairness analysis: A causal toolkit for fair machine learning Year: (2024)
Ref_id:b13 Title: Single world intervention graphs: a primer Year: (2013)
Ref_id:b14 Title: What Counterfactuals Can Be Tested Year: (2007)
