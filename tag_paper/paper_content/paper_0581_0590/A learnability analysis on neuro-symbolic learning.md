Title: A Learnability Analysis on Neuro-Symbolic Learning
Abstract: This paper presents a comprehensive theoretical analysis of the learnability of neuro-symbolic (NeSy) tasks within hybrid systems. We characterize the learnability of NeSy tasks by their derived constraint satisfaction problems (DCSPs), demonstrating that a task is learnable if and only if its corresponding DCSP admits a unique solution. Under mild assumptions, we establish the sample complexity for learnable tasks and show that, for general tasks, the asymptotic expected concept error is controlled by the degree of disagreement among DCSP solutions. Our findings unify the characterization of learnability and the phenomenon of reasoning shortcuts, providing theoretical guarantees and actionable guidance for the principled design of NeSy systems.

Section: Introduction
Neuro-symbolic learning (NeSy) seeks to integrate data-driven learning with knowledge-driven reasoning within a unified framework (Hitzler & Sarker, 2022;Marra et al., 2024). State-of-the-art NeSy approaches predominantly employ hybrid systems that map input queries x to concepts ẑ via a learning model, subsequently utilizing a symbolic system KB to deduce the final answer ŷ. Feedback from KB-such as pseudo-labels in abductive learning (ABL) (Zhou, 2019;Dai et al., 2019) or weighted model counting in DeepProbLog (Manhaeve et al., 2018(Manhaeve et al., , 2021a)-guides further learning. This paradigm has found broad application in domains including puzzle solving, code generation, and autonomous path planning (Jiao et al., 2024;Li et al., 2024;Hu et al., 2025a).
A central challenge in NeSy is that systems are typically trained end-to-end in a weakly supervised manner, relying solely on (x, y) pairs, with the underlying concepts z remaining unobserved. The objective is to learn a model f : X → Z that generalizes effectively, minimizing the concept risk: Under what conditions can the concept risk be minimized through empirical risk minimization over the NeSy risk, given a finite sample set as its size approaches infinity?
Recently, Marconato et al. (2023b) identified the reasoning shortcut problem, wherein a reasoning shortcut (RS) refers to a concept distribution that achieves maximal likelihood on training data yet deviates from the true underlying concept distribution. This phenomenon is closely related to the statistical learnability of NeSy systems, as models may minimize empirical NeSy risk without necessarily minimizing the concept risk. While several approaches have been proposed to address reasoning shortcuts Marconato et al. (2023aMarconato et al. ( , 2024)), a rigorous theoretical framework connecting RSs to the statistical learnability of NeSy tasks remains underexplored.
To address this gap, we analyze the learnability of the NeSy task within the probably-approximatelycorrect (PAC) framework (Valiant, 1984), focusing on restricted hypothesis spaces (cf. section 3.3).
Our key insight is to formulate NeSy tasks as derived constraint satisfaction problems (DCSPs): a task is learnable if and only if its DCSP has a unique solution.
We further introduce disagreement d among DCSP solutions as a finer-grained measure of uncertainty. For learnable tasks, we establish the sample complexity 1 /κ • log (|B|/ϵ), where κ, |B| are task-specific and ϵ is the desired concept error (cf. theorem 3.6). For general NeSy tasks, we derive an upper bound on the expected concept error, showing it is bounded by d/L, where L is the concept space size (cf. theorem 3.7). Moreover, with the DCSP perspective, we find that aggregating unlearnable tasks in a multi-task learning manner reduces the degree of ambiguity, thereby enhancing overall task learnability.
Our analysis aligns with the RS phenomenon (Marconato et al., 2023a,b), wherein models may achieve low empirical risk yet incur high concept risk. The existence of deterministic RSs corresponds to the presence of multiple DCSP solutions. We show that concept error correlates more strongly with solution disagreement than with the number of solutions (cf remark 1), underscoring disagreement d as a more informative indicator.
In summary, our contributions are as follows:
• We establish a rigorous theoretical foundation for NeSy learnability within the hybrid systems paradigm and derive both sample complexity and asymptotic bounds on concept error.
• The DCSP framework proposed facilitates intuitive analysis of learnability using CSP solvers, while solution disagreement provides a detailed perspective on concept error analysis.
• Empirical studies are conducted to validate our theoretical findings.
The remainder of the paper is organized as follows: Section 2 reviews the background and related work on NeSy. Section 3 presents our theoretical analysis of NeSy task learnability. Section 4 details empirical validations of our theoretical results. Section 5 discusses limitations, and section 6 concludes the paper.
this section cite: ['b44', 'b65', 'b10', 'b37', 'b24', 'b32', 'b43', 'b53']

Section: Preliminaries
This section first introduces the problem setup of neuro-symbolic learning, followed by a presentation of the two principal neuro-symbolic methods: probabilistic neuro-symbolic learning and abductive learning. Subsequently, we discuss the most related works, including those on reasoning shortcuts and theoretical analyses of NeSy.
this section cite: []

Section: Problem Setup
A typical hybrid neuro-symbolic system consists of two components: a machine learning model (e.g., a neural network) and a logical reasoning model (e.g., a first-order logic solver). The learning model f : X → Z maps an instance x (e.g., image, text, or audio) from the input space X to an intermediate concept z (e.g., primitive facts or predicates) within the symbol space Z, where |Z| = L. The reasoning model KB consists of rules over the concept space and can be implemented using any logic-based system, such as ProbLog (De Raedt et al., 2007) or answer set programming (Dimopoulos et al., 1997). Assume that a labeling function g : X → Z exists such that z = g(x). The learning model, belonging to a hypotheses family F, is parameterized by θ, and p θ (•) represents the likelihood estimated by the model, where f (x) = arg max z∈Z p θ (z | x).
During inference, the learning model f accepts multiple instances as a sequence x = (x 1 , . . . , x m ) and outputs a sequence of concepts ẑ = (ẑ 1 , . . . , ẑm ). The output ẑ is then passed to the reasoning model KB, which infers the final label y ∈ Y through logical entailment, i.e., ẑ ∧KB |= y. To simplify the inference process of KB, we represent it by a logical forward operator σ(•) such that σ( ẑ) = y. In a standard neuro-symbolic learning setup (Dai et al., 2019;Manhaeve et al., 2021a;Li et al., 2023;Marra et al., 2024), the training data {(x i , y i )} N i=1 are sampled from data distribution D = (X m , Y). Therefore, a neuro-symbolic task can be formally defined as a triple T = ⟨X , Y, KB⟩.
Example 1 (Addition). The input x ∈ X 2 , where X denotes digit images. The concepts Z consist of digits from 0 to 9. The data takes the form ( , ) → 1, with the logical forward function σ(•) := SUM(•, •) . The label space Y is defined as the addition results, i.e., from 0 to 18.
Note that when all final labels y are identical (e.g., z ∧ KB |= ⊤), as in the case of code generation where all code must satisfy a syntax constraint (Jiao et al., 2024), the final label y can be omitted for simplicity. The analysis presented in this paper can be easily adapted to such cases.
The success of NeSy systems highly depends on the recognition of intermediate concepts. To evaluate the concept-level performance, we define the concept risk as follows:
R 0/1 (f ; g) = E x [I (f (x) ̸ = g(x))] .(1)
For simplicity, we omit g and denote (1) as R 0/1 (f ). However, optimizing (1) is challenging due to the lack of supervision regarding the intermediate concept.
this section cite: ['b11', 'b12', 'b10', 'b31', 'b44', 'b24']

Section: Neuro-Symbolic Methods
To minimize the concept risk (1), the key idea of current NeSy methods (Manhaeve et al., 2018;Dai et al., 2019) is to optimize the neuro-symbolic risk as a surrogate, which aims to minimize the discrepancy between the learning and reasoning models.
R NeSy (f ) = E (x,y) [I (f (x) ∧ KB ̸ |= y)] .(2)
The optimization process is to select the optimal f * ∈ F that minimizes the NeSy risk; that is:
f * = arg min f ∈F R NeSy (f ).(3)
Probabilistic Neuro-Symbolic Learning. Probabilistic neuro-symbolic learning (PNL, Manhaeve et al. 2021b) methods adopt reasoning models via probabilistic logic programming, such as DeepProblog (Manhaeve et al., 2018(Manhaeve et al., , 2021a)), NeurASP (Yang et al., 2020), and Scallop (Li et al., 2023). Since the NeSy risk (2) is non-continuous, PNL aims to minimize the following objective:
-E (x,y) p [y | x; f, KB] .(4)
Reformulating (4), we can express the objective as follows:
-E (x,y) log z I(z ∧ KB |= y) • p [z | x; f, KB] .(5)
Equation ( 5) is referred to as the probabilistic neuro-symbolic learning risk, denoted as R PNL (f ).
The key operation for calculating the PNL risk is z I(z∧KB |= y)•p [z | x; f, KB], also well-known as weighted model counting (WMC), which requires enumerating all possible worlds that satisfy the constraints of the symbolic system. This operation can be performed using various approaches, such as ProbLog (De Raedt et al., 2007), answer set programming Dimopoulos et al. (1997), and so on. However, in general, the computational complexity of WMC is #P (Maene et al., 2024), which makes PNL methods challenging to scale.
Abductive Learning. Unlike PNL methods, abductive learning methods (ABL) (Dai et al., 2019;Huang et al., 2021;Hu et al., 2025a) infer the most plausible concepts through abductive reasoning and use them to update the model. The objective of ABL is to minimize the risk:
R ABL (f ) = -E (x,y) log (p [y, z | x; f, KB]) ,(6)
where z = min z∈A(y) Score(z, f (x)) represents the most likely candidate in the abduction set. The abduction set A(y) includes all possible concepts z that satisfy the constraints of KB and have a non-zero measure, i.e., p[z] > 0. The score function measures the alignment between a candidate z and the model's prediction f (x). For instance, Dai et al. (2019) use the Hamming distance.
ABL enhances computational efficiency by concentrating on the most plausible candidates, thereby avoiding the enumeration of all possible worlds. However, the inherent ambiguity of abduction can lead to incorrect candidate selection, introducing bias into the learning process (He et al., 2024b).
We now present a unified perspective: both PNL and ABL approaches are capable of effectively optimizing (2), thereby allowing the analysis to be uniformly applicable to both methods. Accordingly, we formally state the following theorem, with its proof provided in section B.1.
Theorem 2.1. A minimizer of R PNL or R ABL is also a minimizer of R NeSy . For each surrogate R s ∈ {R PNL , R ABL }, we have:
arg min f ∈F R s (f ) ⊆ arg min f ∈F R NeSy (f ),
this section cite: ['b37', 'b10', 'b37', 'b63', 'b31', 'b11', 'b12', 'b34', 'b10', 'b22', 'b10']

Section: Related Works
Here we review related works below that focus on reasoning shortcuts and the theoretical analysis of neuro-symbolic learning. A more comprehensive review of related works is available in section A.
this section cite: []

Section: Reasoning Shortcuts.
A reasoning shortcut (RS) is formally defined as a distribution that maximizes the training likelihood while deviating from the true concept distribution (Marconato et al., 2023b). To address RS, a variety of mitigation strategies have been proposed, including continual learning paradigms that structure tasks sequentially to promote knowledge retention (Marconato et al., 2023a), entropy regularization to encourage more faithful concept representations (Marconato et al., 2024), and the development of dedicated benchmarks for systematic evaluation (Bortolotti et al., 2024). Recent works have also introduced new metrics and theoretical analyses to better characterize and quantify shortcut risks (Yang et al., 2024). Despite these advances, a comprehensive theoretical understanding of RS remains limited, particularly regarding the specific concept error analysis under the statistical learning framework. Our work builds upon these foundations by providing a unified theoretical framework that elucidates the learnability of neuro-symbolic tasks, offering new insights into this phenomenon.
this section cite: ['b43', 'b6', 'b62']

Section: Theoretical Analyses.
Prior theoretical frameworks, such as multi-instance partial label learning with the M -unambiguity condition (Wang et al., 2023), provide concept error bounds based on the VC-dimension. However, this analysis assume repetitive input patterns, such as [z, z, . . . ], thereby limiting the applicability to real-world scenarios that require heterogeneous predicates and facts as logical inputs. Tao et al. (2024) examine scenarios where randomly selecting abduction candidates results in a consistent optimization objective within the ABL framework. They formulate the learning process as a weakly supervised learning problem and analyze its consistency through a probabilistic matrix Q. However, construction of such a matrix Q requires full knowledge of the underlying concept sequence distribution, which is not easily obtainable. Additionally, Yang et al. (2024) introduce a shortcut risk metric R s to quantify the discrepancy between true risk and surrogate risk. While they establish error bounds for this metric, a low shortcut risk does not necessarily guarantee a low concept error. Thus, a comprehensive theoretical analysis of the concept error remains lacking.
In summary, our work advances the field by providing a unified and comprehensive theoretical framework for analyzing the learnability of neuro-symbolic learning tasks. We establish necessary and sufficient conditions for learnability under mild assumptions, introduce a constraint satisfaction perspective that enables systematic verification, and derive meaningful error bounds even in unlearnable cases.
this section cite: ['b59', 'b51', 'b62']

Section: Learnability Analysis
In this section, we examine the learnability of neuro-symbolic (NeSy) tasks, specifically whether the concept risk can be minimized via empirical risk minimization (ERM) over the NeSy risk as the sample size approaches infinity. While learnability is attainable in certain scenarios, it is not universally guaranteed. To elucidate the underlying reasons, we conduct a rigorous learnability analysis to address the question: Which classes of NeSy tasks are learnable?
Analogous to the standard probably approximately correct (PAC) learning framework (Valiant, 1984), we formalize the learnability of a NeSy task as follows:
Definition 3.1. Let N denote the size of samples drawn i.i.d. from D, T represent a NeSy task, and F denote the hypothesis space. We say that T is learnable if: for any 0 < ϵ, δ < 1 and distribution D, there exists an algorithm A and an integer N ϵ,δ such that, whenever N ≥ N ϵ,δ , the selected hypothesis f = A(D) satisfies p[R 0/1 ( f ) ≤ ϵ] ≥ 1 -δ. Otherwise, we say that it is unlearnable.
Our analysis focuses on the ERM algorithm as A, given its proven efficacy in common learning settings such as supervised classification and regression, where a problem is learnable if and only if it is learnable by ERM (Blumer et al., 1989;Alon et al., 1997).
this section cite: ['b53', 'b5', 'b1']

Section: Restricted Hypothesis Space
Unlike conventional supervised learning, which learns a hypothesis x → y from pairs (x, y), a NeSy task seeks a mapping x → z from pairs (x, y), where z denotes latent symbolic variables constrained by the reasoning module. Ideally, each y determines a unique z; that is, ∀y ∈ Y, |A(y)| = 1. In this case the task reduces to a standard learning problem (Vapnik, 1999). In practice this is rarely true: many y admit multiple feasible solutions z 1 , . . . , z k , making the task inherently ambiguous. We call the task ambiguous if there exists y ∈ Y with |A(y)| ≥ 2. In statistical learning theory, the complexity of a hypothesis space is characterized by the notion of shattering.
Definition 3.2 (Shattering). A hypothesis space F shatters a finite set X = {x 1 , . . . , x m } with respect to a label space Z if, for every labeling function ℓ : X → Z, there exists a hypothesis f ∈ F such that f (x i ) = ℓ(x i ) for all x i ∈ X.
Using this notion, we obtain the following proposition.
Proposition 3.3. For an ambiguous NeSy task T , if the hypothesis space F shatters the task, then there exists a hypothesis f * that minimizes R NeSy but does not minimize R 0/1 .
The proof is provided in section B.2. Proposition 3.3 suggests that ambiguous NeSy tasks may be unlearnable when the hypothesis space is very complex, such as nearest neighbor, whose Vapnik-Chervonenkis dimension is infinite (Karacali & Krim, 2003), or deep neural networks (Bartlett & Maass, 2003) without any regularization terms. This issue arises due to overfitting caused by the high memorization capacity of models (Zhang et al., 2021). Previous studies emphasize the importance of constraining the hypothesis space in NeSy tasks (Yang et al., 2024). For example, pre-training models or self-supervised learning methods (Sohn et al., 2020) have been shown to promote clustering properties in neural networks, further enhancing generalization performance.
Consider a scenario where a pre-trained model satisfies a clustering property (Huang et al., 2021), meaning that instances representing the same concept are grouped together in feature space. In the ambiguous task described in example 1, if the model correctly processes a key sample such as SUM( , ) = 0, it can reliably identify 0. This, in turn, simplifies subsequent tasks. For example, once the model recognizes SUM( , ) = 1, it can correctly identify 1. By iteratively applying this process, the model can learn to recognize all relevant concepts despite initial ambiguity.
The above process highlights the need to restrict the hypothesis space for the learning system. This hypothesis space ensures consistent mappings between concepts and labels. Let F * be a restricted hypothesis space which ensures that instances with the same label correspond to the same concept, and vice versa. Given the labeling function g, formally, for any f ∈ F * :
∀x 1 , x 2 ∈ X , g(x 1 ) = g(x 2 ) ⇐⇒ f (x 1 ) = f (x 2 ).
this section cite: ['b55', 'b26', 'b3', 'b64', 'b62', 'b49', 'b22']

Section: Derived Constraint Satisfaction Problem
The restricted hypothesis space implicitly partitions the raw input space X into L clusters. We use ⟨x⟩ i to denote the cluster {x | x ∈ X , f (x) = i}. The learning process is to establish a mapping between the clusters {⟨x⟩ 1 , . . . , ⟨x⟩ L } and Z that minimizes the NeSy risk. This process inherently transforms the NeSy learning problem into a constraint satisfaction problem (CSP). In this paper, we refer to it as a derived CSP (DCSP).
The derived constraint satisfaction problem for a NeSy task T is defined as a triple ⟨V, D, C⟩, where: V = {V 1 , . . . , V L } are the variables, D = {D 1 = Z, . . . , D L = Z} are the domains, and C = {C 1 , . . . , C N } are the constraints. Each V i corresponds to a mapping from ⟨x⟩ i to a concept label. For convenience, we slightly abuse notation by letting V(x) denote a mapping from an input sequence to the corresponding concept sequence determined by the mapping set V. Each C j corresponds to a constraint (x j , y j ), e.g., V(x j )∧KB |= y j . Solving the DCSP is to find a consistent assignment I that satisfies all constraints.
A DCSP solution I corresponds to an assignment of values to variables, expressed as
I = {(V 1 , v 1 ), . . . , (V L , v L )},
where each v i is the value assigned to the variable V i . For simplicity, we denote the solution as I = (v 1 , . . . , v L ) by omitting the variables. Here we only discuss the case when the DCSP has solution; Otherwise, the learning model will inevitably conflict with the background KB.
this section cite: []

Section: Conditions of Learnability
In general, the solution to a DCSP may not be unique, i.e., multiple distinct solutions may exist. We denote the solution space as S = {I 1 , . . . , I k }. To characterize the relationships among these solutions, we define an operation Union(), which captures the common assignments among the solutions. When the input set consists of a single element, this operation simply returns that element. The DCSP solution disagreement d quantifies the inconsistency among all solutions:
d = L -|Union(S)|.
The disagreement d measures the number of variables whose values differ across the solutions in S. If d = 0, i.e., |S| = 1, there is a unique solution. In this case, the optimal hypothesis can be identified by minimizing the NeSy risk. Formally, we have:
Lemma 3.4. For a NeSy task T , if the DCSP solution disagreement d = 0, then the NeSy risk is consistent with the concept risk. Formally, for any f ∈ F:
R NeSy (f ) → 0 ⇐⇒ R 0/1 (f ) → 0.
Proof Sketch. The direction from the right-hand side to the left-hand side is straightforward; here, we focus on proving the reverse direction. We demonstrate this by showing that if the concept risk is non-zero, then the NeSy risk cannot be zero (contraposition). If the concept risk is non-zero, there must be at least one misclassified instance where f assigns an incorrect label. Given that the DCSP solution is unique and there is no disagreement (i.e., d = 0), any such misclassification directly results in a non-zero NeSy risk. Therefore, if the NeSy risk is zero, it follows that the concept risk must also be zero.
The detailed proof is in section B.3. To proceed, we introduce the following mild assumption.
Assumption 3.5. The set of possible concept sequences, B = y∈Y A(y), where A(y) is the set of valid concept combinations for label y, has finite cardinality; and the probability of sampling a concept sequence is at least κ > 0
With this assumption, we formally present the main result of this paper as follows.
Theorem 3.6. For a neuro-symbolic task T with a restricted hypothesis space F * , learnability is determined by the following conditions:
• If the derived constraint satisfaction problem has a unique solution, the task is learnable. Specifically, the concept error is bounded by ϵ, provided that the sample size N satisfies:
N > 1 κ • log (|B|/ϵ) .
• Otherwise, the task is unlearnable.
The proof is in section B.4 . Theorem 3.6 establish that a NeSy task T is learnable if and only if the DCSP solution is unique, i.e., disagreement d = 0 . Conversely, if the DCSP has multiple solutions (i.e., d ≥ 1), the task is unlearnable, implying that concept error remains unavoidable regardless of additional training data.
Building upon the concept of DCSP solution disagreement, we derive a more general theorem offering deeper insights into learning errors in a restricted hypothesis space F * . As the sample size approaches infinity, the hypotheses learned via ERM asymptotically converge to: F * ERM = arg min f ∈F * R NeSy (f ). The average error of the ERM result, denoted by E * , is the expected concept risk of an arbitrarily selected hypothesis:
E * = E f ∈F * ERM [R 0/1 (f )] . Theorem 3.7. The average error E * is bounded by: E * ≤ d L .
The proof is in section B.5. Theorem 3.7 provides an asymptotic error analysis for NeSy tasks, indicating that as the DCSP solution disagreement d increases, the upper bound of the concept error also increases. Revealing that the disagreement d is crucial to the learnability of NeSy tasks.
this section cite: []

Section: Examples
Table 1: Examples of (un)learnable tasks.
Learnable Addition y = z 1 + z 2 Multiplication y = z 1 × z 2 Unlearnable Exclusive OR y = z 1 ⊕ z 2 Modular Addition y = (z 1 + z 2 ) mod k
Here, we present examples to better understand the learnability conditions of a NeSy task. To illustrate the distinction between learnable and unlearnable tasks, we use digital images as input data. We model the data as x = (x 1 , x 2 ) ∈ X 2 , where X represents the space of digit images (e.g., { , , . . . }). The intermediate concept space Z and the label space Y depend on the specific knowledge base. Table 1 summarizes these examples,where in modular addition task 2 ≤ k ≤ 10.
For the XOR task (d/L = 1), interchanging the concepts 0 and 1, i.e., → 0, → 1 and vice versa, minimizes the NeSy risk. For the modular addition task (k = 9, d/L = 0.2), swapping the mappings of 0 and 9, i.e., → 0, → 9 and vice versa, minimizes the NeSy risk.
Remark 1. A direct implication from theorem 3.7 is that the expected concept error does not increase monotonically with the number of DCSP solutions but is instead with the disagreement d. For example, in the modular addition task: (i) When k = 8, there are 4 solutions, with d/L = 0.4 ; However, (ii) when k = 10, there are only 2 solutions, yet d/L = 1 . This occurs because the significant disagreement between the two solutions leads to an unbounded worst-case concept error.
this section cite: []

Section: Aggregation of Unlearnable Tasks
Certain NeSy tasks are inherently unlearnable because they admit multiple solutions to their DC-SPs, resulting in ambiguity. This ambiguity cannot be resolved by increasing data or improving the learning algorithm, as it stems from intrinsic task properties. Interestingly, however, such unlearnable tasks may become learnable when combined under a multi-task learning paradigm. The key insight is that tasks can mutually constrain each other, reducing ambiguity.
Consider two unlearnable tasks, T 1 and T 2 , with their solution spaces S 1 and S 2 , where |S 1 | ≥ 2 and |S 2 | ≥ 2. In a multi-task learning setting, the combined task requires satisfying constraints from both tasks at the same time, creating the solution set S agg = S 1 ∩ S 2 . For the combined task to become learnable, two key conditions must hold: (1) concept space overlap: Z 1 ∩ Z 2 ̸ = ∅; and (2) reduced DCSP disagreement:
d agg = |Z 1 ∪ Z 2 | -|Union(S agg )| < min(d 1 , d 2 ).
This reduction of the solution space reduces ambiguity and may lead to a unique solution, making the combined task learnable. Therefore, by using the mutual constraints from overlapping solution spaces, combining unlearnable tasks in an aggregation framework can enable learnability. From the perspective of DCSP, we can formally state the corollary as follows:
Corollary 3.8. NeSy tasks become learnable in an aggregation framework if combining their DCSPs results in a unique solution.
this section cite: []

Section: Empirical Study
To empirically validate the theoretical results, we conducted a series of experiments, including arithmetic tasks shown in table 1 and BDD-OIA Xu et al. (2020) Setup Manhaeve et al. (2018) proposed the digit addition task by incorporating the handwritten MNIST (LeCun et al., 1994) and predefined addition rules. We extend the setup by including KMNIST (Clanuwat et al., 2018), CIFAR10 Krizhevsky (2009), and SVHN (Netzer et al., 2011), mapping class indices to digits, e.g., CIFAR-10 classes (airplane = 0, . . .) , and enriching the background knowledge as depicted in table 1. The learning model for MNIST and KMNIST is LeNet (LeCun & Bengio, 1998), while ResNet50 (He et al., 2016) is used for CIFAR10 and SVHN. Besides that, we also adopt BDD-OIA from Bortolotti et al. (2024), which is a multi-label autonomous driving task for studying RSs in real-world, high-stakes scenarios. All experiments were conducted five times with different random seeds. Details can be seen in section C.
10 1 10 2 10 3 10 4 Sample size (log scale) 0.0 0.2 0.4 0.6 0.8 1.0 Accuracy #Sols:1 Addition 10 1 10 2 10 3 10 4 Sample size (log scale) #Sols:2 Mod Addition k = 9 10 1 10 2 10 3 10 4 Sample size (log scale) #Sols:4 Mod Addition k = 8 10 1 10 2 10 3 10 4 Sample size (log scale) #Sols:16 Mod Addition k = 6 10 1 10 2 10 3 10 4 Sample size (log scale) #Sols:144 Mod Addition k = 4 10 1 10 2 10 3 10 4 Sample size (log scale) #Sols:28800 Mod Addition k = 2 Concept Acc Reasoning Acc Asymptotic Bound 10 1 10 2 10 3 10 4 Sample size (log scale) 0.0 0.2 0.4 0.6 0.8 1.0 Accuracy #Sols:1 Addition 10 1 10 2 10 3 10 4 Sample size (log scale) #Sols:2 Mod Addition k = 9 10 1 10 2 10 3 10 4 Sample size (log scale) #Sols:4 Mod Addition k = 8 10 1 10 2 10 3 10 4 Sample size (log scale) #Sols:16 Mod Addition k = 6 10 1 10 2 10 3 10 4 Sample size (log scale) #Sols:144 Mod Addition k = 4 10 1 10 2 10 3 10 4 Sample size (log scale) #Sols:28800 Mod Addition k = 2 Concept Acc Reasoning Acc Asymptotic Bound Method To effectively optimize the NeSy risk (2), we adopt the following surrogate (cf. proof in section B.1):
-E (x,y) log   z∈N (y) p [y, z | x; f, KB]   ,(7)
which is flexible, where N (y) ⊆ A(y) represents several valid candidates for the final answer y. By restricting the size of N (y) from the entire set A(y) to the most likely candidate z, we achieve a balance between PNL and ABL, and we set the size of N (y) is min (16, |A(y)|). The implementation is based on the code of He et al. (2024b). For brevity, detailed experiments on PNL and ABL are provided in section C. We empirically evaluate the learnability of NeSy tasks based on theorem 3.6, focusing on two key aspects: (i) validating that minimizing the NeSy risk consistently minimizes the concept risk for learnable tasks, and (ii) examining how DCSP solution disagreement affects learnability.
this section cite: ['b37', 'b30', 'b9', 'b28', 'b45', 'b17', 'b6']

Section: Empirical Analysis on Learnability
(i) Validation of learnable tasks. We first validate the learnability conditions (cf. theorem 3.6) by examining addition and multiplication tasks (cf. table 1). Solving the DCSP shows that both tasks are learnable, and their learnability remains unaffected by increases in digit size (e.g., from PROD( , ) = 2 to PROD( , ) = 200). The raw dataset in figure 2 is MNIST, and additional results for other datasets are in the appendix. We further substantiate learnability by examining tasks with varying digit sizes, ranging from one to four digits. As depicted in figure 2, the results confirm that: (a) Optimization of the surrogate risk (7) effectively minimizes the NeSy risk. (b) For learnable tasks, a good minimizer of the NeSy risk also serves as a reliable minimizer of the concept risk.
(ii) Impact of DCSP solution disagreement. We further investigate how disagreement in DCSP solutions impacts learnability. According to theorem 3.7, the asymptotic error is bounded by the ratio of DCSP solution disagreement d to the size of the concept space L. Experiments involving addition and modular addition tasks with varying modular bases k reveal that altering the knowledge base changes the DCSP solution space, directly influencing learnability. For clarity, we plot the asymptotic accuracy bound for each task, i.e., 1 -d/L, showing that higher disagreement results in a lower bound line (green). As shown in figure 1: (a) Tasks with a unique DCSP solution are learnable; (b) Tasks with high DCSP disagreement struggle to achieve low concept risk, even as the sample size increases.
this section cite: []

Section: Predicted Actual
0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.83 0.00 0.00 0.00 0.00 0.00 0.00 0.16 0.99 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.03 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.97 0.99 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.99 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.98 0.98 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02
this section cite: []

Section: Mod Addition k = 2
Predicted 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.17 0.00 0.00 0.83 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.99 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.00 0.00
this section cite: []

Section: Mod Addition k = 3
Predicted 0.46 0.01 0.00 0.00 0.00 0.06 0.29 0.00 0.06 0.12 0.00 0.22 0.00 0.00 0.59 0.02 0.00 0.14 0.03 0.00 0.00 0.00 0.71 0.00 0.00 0.15 0.00 0.00 0.12 0.00 0.01 0.01 0.00 0.71 0.00 0.02 0.14 0.00 0.02 0.08 0.01 0.18 0.00 0.00 0.59 0.02 0.02 0.11 0.07 0.01 0.01 0.01 0.00 0.00 0.00 0.62 0.02 0.00 0.33 0.01 0.19 0.02 0.00 0.00 0.00 0.06 0.59 0.00 0.08 0.06 0.00 0.12 0.00 0.00 0.57 0.02 0.00 0.22 0.06 0.00 0.01 0.01 0.00 0.00 0.00 0.62 0.01 0.01 0.34 0.00 0.16 0.07 0.00 0.00 0.05 0.03 0.22 0.04 0.07 0.37
Aggregation(2, 3) Epoch 0.4 0.6 0.8 1.0 Accuracy Mod Addition k = 2 Epoch Mod Addition k = 3 Epoch Reasoning Acc Concept Acc Aggregation(2, 3)
this section cite: []

Section: Predicted Actual
1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.17 0.00 0.00 0.83 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.99 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.00 0.00
this section cite: []

Section: Mod Addition k = 3
Predicted 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.16 0.00 0.00 0.00 0.84 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99
this section cite: []

Section: Mod Addition k = 4
Predicted 0.98 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.98 0.00 0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.96 0.01 0.00 0.01 0.00 0.01 0.01 0.00 0.00 0.00 0.00 0.99 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 0.00 0.00 0.00 0.00 0.00 0.01 0.00 0.01 0.00 0.00 0.95 0.00 0.00 0.01 0.01 0.00 0.00 0.01 0.00 0.00 0.01 0.97 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.01 0.00 0.00 0.98 0.00 0.00 0.01 0.00 0.00 0.00 0.01 0.02 0.00 0.00 0.95 0.01 0.02 0.00 0.00 0.00 0.02 0.01 0.00 0.00 0.00 0.94 Aggregation(3, 4) In summary, our empirical analysis confirms the theoretical learnability conditions by demonstrating that minimizing the NeSy risk reliably minimizes the concept risk for learnable tasks. Furthermore, tasks with lower disagreement exhibit better learnability, while those with high disagreement suffer from ambiguity due to multiple conflicting solutions.
4.2 Further Evaluation on a Realistic Task: BDD-OIA  (Manhaeve et al., 2021a) 44.69±0.19 0.00±0.00 ABL (Dai et al., 2019) 75.16±0.12 66.92±5.86 A 3 BL (He et al., 2024b) 60.91±0.01 88.37±0.00
We further extend our experiments to the realistic BDD-OIA task, a multi-label autonomous driving benchmark in real-world, high-stakes scenarios. The knowledge base KB encodes, for example, that it is unsafe to move forward when pedestrians are present, using a set of 21 binary concepts indicating various obstacles on the road. The constraints specify conditions for proceeding (green light ∨ follow ∨ clear ⇒ forward), stopping (red light ∨ stop sign ∨ obstacle ⇒ stop), turning left and right, and relationships between actions (e.g., stop ⇒ ¬forward). In this task, there are 74,240 DCSP solutions, and the disagreement among these solutions is 15, indicating a typically unlearnable problem under our criterion. We evaluate several methods on this task: LTN and DeepProbLog are implemented using the rsbench codebase (Bortolotti et al., 2024), whereas ABL and A 3 BL use their official implementations.
this section cite: ['b10', 'b6']

Section: Aggregation of Unlearnable NeSy Tasks
With the DCSP framework, we find that certain NeSy tasks are inherently unlearnable because their DCSPs admit multiple solutions, resulting in inherent ambiguity. However, when combined in an aggregation framework within a multi-task learning setting, such tasks may become learnable by enforcing mutual consistency, as shown in corollary 3.8. We evaluate corollary 3.8 using mod addition tasks with mod bases k 1 and k 2 under two specific configurations: an unlearnable aggregation (k 1 = 2, k 2 = 3) and a learnable aggregation (k 1 = 3, k 2 = 4). For k = 2, 3, 4, the degree of DCSP solution disagreement d is 10. The experiments in figure 3 are based on the raw MNIST dataset. Additional details and experiments are provided in section C.2.3.
In the top of figure 3, the unlearnable case (k 1 = 2, k 2 = 3) shows that while the aggregation narrows the solution space, reducing the disagreement d to 8, it does not converge to a unique solution, and the task remains unlearnable. In the bottom of figure 3, the learnable case (k 1 = 3, k 2 = 4) illustrates that both tasks initially admit multiple DCSP solutions, causing reasoning accuracy to exceed concept accuracy, as shown in figure 3. Through the aggregation, the intersection of solution spaces shrinks, with the disagreement d reduced to 0, making the aggregation task learnable.
This experimental result supports corollary 3.8, demonstrating that forming aggregations of different NeSy tasks can enhance learnability by mutually constraining DCSP solution spaces. This finding suggests that collecting tremendous NeSy tasks and jointly learning them in an aggregation manner could improve the learnability and potentially introduce a "scaling law" (Kaplan et al., 2020) in the NeSy domain.
this section cite: ['b25']

Section: Limitations and Future Directions
This paper focuses exclusively on hybrid neuro-symbolic systems, e.g., probabilistic neuro-symbolic and abductive learning methods. Thus the findings may not directly extend to other types of neurosymbolic methods. The analysis of this study relies on a restricted hypothesis space, which is inherently satisfied by models such as neural networks equipped with manifold regularization (Belkin et al., 2006) or self-supervised pretraining (Liu et al., 2021). However, extending the framework to encompass more general hypothesis spaces without requiring this specific property remains an open challenge.
Future work may involve a deeper investigation into extending the learnability framework to encompass a broader range of NeSy systems. Additionally, exploring the learnability of the semisupervised case of NeSy tasks, where some training examples are supervised for intermediate concepts, could be an interesting direction. Developing practical strategies for constructing effective task aggregations aslo represents a promising avenue for improving learnability in many scenarios.
Moreover, because solving CSPs is NP-hard in general, the DCSP framework may face computational scalability challenges for large-scale knowledge bases. Future directions include approximation methods such as sampling-based estimation (e.g., uniform sampling Heradio et al. 2020), incorporating practical heuristics (e.g., exploiting symmetries in the knowledge base), and decomposing complex knowledge bases into simpler sub-tasks with tractable CSPs (Hu et al., 2025b).
this section cite: ['b4', 'b33']

Section: Conclusion
We establish that a neuro-symbolic task is learnable if and only if the derived constraint satisfaction problem (DCSP) has a unique solution. This conclusion is consistent with previously found reasoning shortcuts problem. With the DCSP framework, we can conduct a comprehensive analysis on sample complexity and concept error based on the disagreement d among these solutions. This framework also implies that forming aggregations of unlearnable tasks reduces the disagreement d, thereby enhancing overall task learnability.
this section cite: []

Section: References
Ref_id:b0 Title: Symbolic Knowledge Injection Meets Intelligent Agents: QoS Metrics and Experiments Year: (2023-06)
Ref_id:b1 Title: Scale-sensitive Dimensions, Uniform Convergence, and Learnability Year: (1997-07)
Ref_id:b2 Title:  Year: ()
Ref_id:b3 Title: Vapnik-Chervonenkis Dimension of Neural Nets. The Handbook of Brain Theory and Neural Networks Year: (2003)
Ref_id:b4 Title: Manifold Regularization: A Geometric Framework for Learning from Labeled and Unlabeled Examples Year: (2006)
Ref_id:b5 Title: Learnability and The Vapnik-Chervonenkis Dimension Year: (1989-10)
Ref_id:b6 Title: A Neuro-Symbolic Benchmark Suite for Concept Quality and Reasoning Shortcuts Year: (2024)
Ref_id:b7 Title: Abductive Learning with Ground Knowledge Base Year: (2021)
Ref_id:b8 Title: Symbolic Knowledge Extraction and Injection with Sub-symbolic Predictors: A Systematic Literature Review Year: (2024-03)
Ref_id:b9 Title: Deep Learning for Classical Japanese Literature Year: (2018)
Ref_id:b10 Title: Bridging Machine Learning and Logical Reasoning by Abductive Learning Year: (2019)
Ref_id:b11 Title: ProbLog: A Probabilistic Prolog and Its Application in Link Discovery Year: (2007)
Ref_id:b12 Title: Encoding Planning Problems in Nonmonotonic Logic Programs Year: (1997)
Ref_id:b13 Title: Knowledge-enhanced Historical Document Segmentation and Recognition Year: (2024)
Ref_id:b14 Title: Neural-Symbolic Learning System: Foundations and Applications Year: (2002)
Ref_id:b15 Title: Reduced Implication-bias Logic Loss for Neuro-symbolic Learning Year: (2024)
Ref_id:b16 Title: Ambiguity-aware Abductive Learning Year: (2024)
Ref_id:b17 Title: Deep Residual Learning for Image Recognition Year: (2016)
Ref_id:b18 Title: Uniform and Scalable SATsampling for Configurable Systems Year: (2020)
Ref_id:b19 Title: Neuro-Symbolic Artificial Intelligence: The State of the Art Year: (2022)
Ref_id:b20 Title: Efficient Rectification of Neuro-Symbolic Reasoning Inconsistencies by Abductive Reflection Year: ()
Ref_id:b21 Title: Curriculum Abductive Learning Year: (2025)
Ref_id:b22 Title: Fast Abductive Learning by Similarity-based Consistency Optimization Year: (2021)
Ref_id:b23 Title: ABLkit: A Python Toolkit for Abductive Learning Year: (2024)
Ref_id:b24 Title: Valid Text-to-SQL Generation with Unification-Based Deep-StochLog Year: (2024)
Ref_id:b25 Title: Scaling Laws for Neural Language Models Year: (2020)
Ref_id:b26 Title: Fast Minimization of Structural Risk by Nearest Neighbor Rule Year: (2003)
Ref_id:b27 Title: A Method for Stochastic Optimization Year: (2015)
Ref_id:b28 Title: Learning Multiple Layers of Features from Tiny Images Year: (2009)
Ref_id:b29 Title: Convolutional Networks for Images, Speech, and Time Series Year: (1998)
Ref_id:b30 Title: The MNIST Database of Handwritten Digits Year: (1994)
Ref_id:b31 Title: Scallop: A Language for Neurosymbolic Programming Year: (2023-06)
Ref_id:b32 Title: Neuro-Symbolic Learning Yielding Logical Constraints Year: (2024)
Ref_id:b33 Title: Self-Supervised Learning: Generative or Contrastive Year: (2021)
Ref_id:b34 Title: On the Hardness of Probabilistic Neurosymbolic Learning Year: (2024)
Ref_id:b35 Title: On the Design of PSyKI: A Platform for Symbolic Knowledge Injection into Sub-symbolic Predictors Year: (2022)
Ref_id:b36 Title: NeSyA: Neurosymbolic Automata Year: (2025)
Ref_id:b37 Title: DeepProbLog: Neural Probabilistic Logic Programming Year: (2018)
Ref_id:b38 Title:  Year: ()
Ref_id:b39 Title: Neuro-Symbolic AI = Neural + Logical + Probabilistic AI Year: (2021)
Ref_id:b40 Title: The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision Year: (2019)
Ref_id:b41 Title: Neuro-Symbolic Continual Learning: Knowledge, Reasoning Shortcuts and Concept Rehearsal Year: ()
Ref_id:b42 Title: Not All Neuro-Symbolic Concepts Are Created Equal: Analysis and Mitigation of Reasoning Shortcuts Year: (2023)
Ref_id:b43 Title: BEARS Make Neuro-Symbolic Models Aware of their Reasoning Shortcuts Year: (2024)
Ref_id:b44 Title: From Statistical Relational to Neurosymbolic Artificial Intelligence: A survey Year: (2024)
Ref_id:b45 Title: Reading Digits in Natural Images with Unsupervised Feature Learning Year: (2011)
Ref_id:b46 Title:  Year: (2022)
Ref_id:b47 Title: Regularizing Deep Networks with Prior Knowledge: A Constraint-based Approach Year: (2021)
Ref_id:b48 Title: Relational Neurosymbolic Markov Models Year: (2025)
Ref_id:b49 Title: FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence Year: (2020)
Ref_id:b50 Title: Integrating Rules and Connectionism for Robust Commonsense Reasoning Year: (1994)
Ref_id:b51 Title: Deciphering Raw Data in Neuro-Symbolic Learning with Provable Guarantees Year: (2024)
Ref_id:b52 Title: Knowledge-based Artificial Neural Networks Year: (1994)
Ref_id:b53 Title: A Theory of The Learnable Year: (1984-11)
Ref_id:b54 Title: Analyzing Differentiable Fuzzy Logic Operators Year: (2022)
Ref_id:b55 Title: An Overview of Statistical Learning Theory Year: (1999)
Ref_id:b56 Title: Modeling PU Learning Using Probabilistic Logic Programming Year: (2023)
Ref_id:b57 Title: Informed Machine Learning -A Taxonomy and Survey of Integrating Prior Knowledge into Learning Systems Year: (2023)
Ref_id:b58 Title: Knowledge-based Stroke Evaluation in Table Tennis Year: (2021)
Ref_id:b59 Title: On Learning Latent Models with Multi-instance Weak Supervision Year: (2023)
Ref_id:b60 Title: A Semantic Loss Function for Deep Learning with Symbolic Knowledge Year: (2018)
Ref_id:b61 Title: Explainable Object-Induced Action Decision for Autonomous Vehicles Year: (2020)
Ref_id:b62 Title: Analysis for Abductive Learning and Neural-Symbolic Reasoning Shortcuts Year: (2024)
Ref_id:b63 Title: Embracing Neural Networks into Answer Set Programming Year: (2020)
Ref_id:b64 Title: Understanding Deep Learning (Still) Requires Rethinking Generalization Year: (2021)
Ref_id:b65 Title: Abductive Learning: Towards Bridging Machine Learning and Logical Reasoning Year: (2019)
