Title: No Soundness in the Real World: On the Challenges of the Verification of Deployed Neural Networks
Abstract: The ultimate goal of verification is to guarantee the safety of deployed neural networks. Here, we claim that all the state-of-the-art verifiers we are aware of fail to reach this goal. Our key insight is that theoretical soundness (bounding the full-precision output while computing with floating point) does not imply practical soundness (bounding the floating point output in a potentially stochastic environment). We prove this observation for the approaches that are currently used to achieve provable theoretical soundness, such as interval analysis and its variants. We also argue that achieving practical soundness is significantly harder computationally. We support our claims empirically as well by evaluating several well-known verification methods. To mislead the verifiers, we create adversarial networks that detect and exploit features of the deployment environment, such as the order and precision of floating point operations. We demonstrate that all the tested verifiers are vulnerable to our new deployment-specific attacks, which proves that they are not practically sound.

Section: Introduction
The formal verification of an artificial neural network produces a mathematical proof that the network has (or does not have) a certain important property. One important property to verify is the adversarial robustness (Szegedy et al., 2014) of classifier networks, where we wish to prove that a subset of inputs are all assigned the same class label.
There is a wide variety of approaches that address this problem (see Section 2). However, these methods invariably focus on the verification of the theoretical model of the network, that is, they seek to characterize the behavior of the full-precision computation. Most solutions carefully address floating point issues as well, but only as an obstacle in the way of verifying the theoretical model, e.g. (Singh et al., 2019a;2018).
At the same time, deployment environments affect numeric computation through hardware and software features, often resulting in stochastic behavior (Schlögl et al., 2023;Villa et al., 2009;Shanmugavelu et al., 2024). This means that the verification of the theoretical model and the verification of the deployed network are different problems. Parallel to our work, a similar observation was also made by (Cordeiro et al., 2025), where this problem is referred to as the implementation gap.
We formally prove that a verifier that is theoretically sound (i.e., bounds the full-precision output correctly) is not necessarily practically sound, that is, it might not bound the actual output correctly in a given deployment environment.
The problem has practical implications. Recently, (Shanmugavelu et al., 2025) demonstrated that adversarial inputs can be generated simply by permuting the order of associative operations in the deployment environment.
We demonstrate a more severe, practically exploitable vulnerability as well: the deployed network is shown to be fundamentally different from the theoretical network because the behavior of the network can be changed arbitrarily with deployment-sensitive backdoors.
Thus, the deployment environment should be a fundamental part of any verification effort because otherwise an attacker can hide potentially harmful behaviors from the verifier if enough information about the deployment environment is available.
We summarize our contributions below:
• We prove that verifiers that are theoretically sound are not necessarily sound for deployed networks
• We demonstrate that deployed networks may differ arbitrarily from the full-precision model, with the help of backdoors triggered by features of the environment
• We complement our theoretical analysis by an empirical evaluation, showing that the state-of-the-art verifiers that are theoretically sound indeed fail to correctly bound the deployments of our backdoored networks 1
this section cite: ['b30', 'b23', 'b32', 'b24', 'b3']

Section: Related work
Verification methods can be classified in various ways (Li et al., 2023;Albarghouthi, 2021;Huang et al., 2020;Liu et al., 2021), but from a safety perspective, the most important classification categorizes algorithms based on their completeness and soundness. A verification algorithm is sound if every property it predicts to be true is true, and complete if it predicts every true property to be true. (Different terminology has also been used in the literature (Bunel et al., 2020)).
Most proposed methods assume a special class of neural networks, typically ReLU networks. Sound (but not necessarily complete) methods, such as (Singh et al., 2018;2019a;Zhang et al., 2018;Xu et al., 2021), are often based on bound propagation (Xu et al., 2020) and linearization. These methods deal with floating point computations as well and bound the full-precision value correctly.
Methods claimed to be both sound and complete often formalize the verification problem as a satisfiability modulo theories (SMT) problem (Katz et al., 2017;Ehlers, 2017) or a mixed-integer linear programming (MILP) model (Dutta et al., 2018;Tjeng et al., 2017) and use SMT or MILP solvers to solve them.
The main disadvantages of these methods include being NPcomplete (Katz et al., 2017), and relying on sophisticated solvers that might include heuristics that break soundness (Zombori et al., 2021). To mitigate the scaling issue, some methods (Tjeng et al., 2017;Singh et al., 2019b) incorporate sound (but not complete) approaches in a preprocessing phase to make the SMT or MILP models more manageable by reducing the number of integer variables.
In addition, a branch-and-bound (BaB) verification framework was also proposed for sound and complete verification (Bunel et al., 2020). The basic idea is to recursively divide the verification problem into simpler subproblems, where inexpensive sound methods can prove robustness. Verifiers based on this framework mainly differ in how they solve the subproblems and how they perform the branching process.
The main advantage of the BaB framework is that the verification process can be performed partially (Xu et al., 2021) or fully (Wang et al., 2021;Palma et al., 2021;Ferrari et al., 2022) on a GPU, significantly increasing the efficiency of the verification. The work of (Zombori et al., 2021) also addresses numeric vulnerabilities in verification, but they exploit the numerical issues of the verification algorithm itself (specifically, the MILP solver). These kinds of exploitable issues are not present in sound verifiers based on bound propagation. Instead, here, we focus on exploiting the discrepancy between full-precision models and their deployment that provably mislead every sound verifier we are aware of. (Jia & Rinard, 2021) also address related issues. Their work can be considered an adversarial attack on verification and not a generic and provable framework we propose. It works only if the deployment and the verification algorithms use a different numerical precision, unlike our constructions.
While we focus on the problems stemming from floating point arithmetic, fixed-point arithmetic should be mentioned as an interesting alternative path where quantization can be implemented with a bounded rounding error (Lohar et al., 2023), allowing for practically sound verification.
this section cite: ['b15', 'b0', 'b10', 'b16', 'b2', 'b26', 'b38', 'b37', 'b36', 'b13', 'b6', 'b5', 'b31', 'b13', 'b40', 'b31', 'b2', 'b37', 'b34', 'b21', 'b7', 'b40', 'b11']

Section: An Intuitive Overview
Here, we discuss the main intuition behind our work in an informal manner, as well as some of the misunderstandings we often encounter.
Main idea. Neural networks are functions that have real valued parameters. This real valued version is the theoretical model. A deployed neural network model, however, is almost always computed using floating point arithmetic, furthermore, deployments are inherently stochastic due to the non-associative nature of floating point arithmetic and parallelization. Our main message is that-while current verification efforts always target the theoretical model-we should target the deployed model. The main reason is that a deployed model is a fundamentally different mathematical object and this difference allows an attacker to design deployment-sensitive behaviors that remain hidden when verifying the theoretical model of the same network without taking the deployment environment into account.
What this work is not. Our work is not about the floating point issues related to the verification of the theoretical model of the neural network. For example, doing the verification in full precision arithmetic does not solve the problem we raise here because that way, one verifies the theoretical model and not the deployed network, except if full precision is used in deployment as well. Also, floating point arithmetic is a problem here not because of rounding errors, but because of its non-associativity and the stochastic nature of the computation depending on the deployment environment.
An illustration. Figure 1 is a conceptual illustration of a binary classifier that was manipulated by an attacker in order to produce malicious behavior in a specific deployment environment that we call adversarial environment. In this illustration, the theoretical model of the network does not show malicious behavior and it is safe for input x as well, that is, the sensitivity domain of x is inside the class of x. In non-adversarial deployments (environments E 1 and E 2 ) the deployed model shows variations relative to the theoretical model and it might or might not be safe for x. However, in the adversarial deployment environment E 3 , the model is unsafe for most inputs by design, as the class predictions are flipped. Note that this behavior is not triggered by specific inputs, instead, it is triggered by the deployment environment. We discuss such malicious networks in Section 7.
this section cite: []

Section: Background and Notations
Here, we introduce the basic notations our study focuses on: the neural network and the verification problem. We also discuss some basic properties of floating point computation briefly.
this section cite: []

Section: Theoretical Model of Neural Networks
Our formulation is inspired by (Ferrari et al., 2022). Let the theoretical neural network be the function f (.; θ) : R n → R m , where θ ∈ R k is a vector of constant real parameters. Throughout the paper, we work with classifier networks, that is, the network f (.; θ) classifies each input x ∈ R n to one of m classes. The class label of x is given by y(x) = arg max i f (x; θ) i , that is, the index of the maximal output value. Note that in many applications the input is often restricted to a subset of R n . Here, we allow every input, without loss of generality.
this section cite: ['b7']

Section: Verification Problem
The robustness verification problem seeks to decide whether every input in some small neighborhood of a fixed input x * has the same class label as x * . More formally, we are given an input domain D ⊆ R n that defines the small environment. D is usually defined by a norm p and a parameter ϵ: D ϵ,p (x * ) = {x : ∥x -x * ∥ p ≤ ϵ}.
We wish to prove that over this domain, every input satisfies a safety property P . In our case, P ⊆ R n captures the fact that the input has the same label as x * . Let us assume the class label of x * is y(x * ). Then, P (x * ) = {x : f (x; θ) y(x * ) = max i f (x; θ) i }. The goal of verification is to prove that the property P holds over the input domain, that is, D ϵ,p (x * ) ⊆ P (x * ).
To make our equations simpler, but without loss of generality, we extend the neural network with an additional affine layer that computes an output vector of dimension m by computing f (x; θ) y(x * ) -f (x; θ) i for every index i. This assumption simplifies the formulation of the verification problem, which now amounts to proving
∀x ∈ D (ϵ,p) (x * ), f (x; θ) ≥ 0,(1)
where, with a slight abuse of notation, we used the same f to denote the modified network. From now on, f will denote this modified function.
this section cite: []

Section: Classification of Verifiers
A verifier is an algorithm to prove the property in Equation (1) for a given x * . The potential outputs of a verifier include true, false and unknown. The verifier is called complete if for every x * , for which Equation ( 1) is true, it returns true. The verifier is called sound if for every x * , for which it returns true, Equation (1) is true.
Sound verifiers have been proposed for special classes of neural networks (see Section 2). However, they are sound only in the sense of bounding f (.; θ) (the full-precision model) from below, which-as we argue in this paper-is not the right target for verification.
this section cite: []

Section: Floating-Point Issues
A floating-point number is represented as s • b e , where s is the signed significand, b is the base (usually, b = 2), and e is the exponent. Available floating-point implementations mainly differ in the number of bits used to represent the significand and the exponent. The IEEE 754-1985 standard introduced the well-known double and single precision formats. In double-precision (or binary64) representation, the exponent and the significand are represented by 11 and 52 bits, respectively. In single-precision (or binary32), the exponent is represented by 8 bits, and the significand by 23 bits. In both cases, 1 bit is used to store the sign.
Since floating-point arithmetic has finite precision, rounding is applied to determine the floating-point result of mathematical expressions, making the arithmetic order-dependent (or non-associative). For the same expression, different rounding modes and operation orders can introduce numerical errors of varying magnitudes. Next, we demonstrate the two most significant issues that arise with floating-point arithmetic through examples.
Non-associative operations. Different orders of operations in a summation can lead to different results. For example 2 53 + 1 -2 53 = 0 when computed in double-precision arithmetic and with rounding towards -∞ (in this fixed order). However, if we change the order to 2 53 -2 53 + 1, the result will be 1 in the same environment.
Precision. Different floating-point representations (such as double and single precision) have a different number of bits to represent the significand. Thus, for example, the result of 2 24 + 1 -2 24 is 0 under single precision, but 1 under double precision.
this section cite: []

Section: Deployed Verification
The effect of different deployment environments have already been investigated in (Schlögl et al., 2023), although not from the point of view of verification. It was shown that different platforms-that differ in number representation, parallelization, hardware, and optimizations during convolution-significantly impact the output of deployed neural networks.
Here, we define the deployed verification problem and show that it differs significantly from the theoretical problem in Equation (1).
this section cite: ['b23']

Section: Deployed Neural Networks
For a theoretical model f (.; θ) and a deployment environment E, let r(.; θ, E) : X → Y be the deployed network.
The notation E fully captures every property of the environment, including every detail of number representation, hardware, software optimizations, operation order, and potentially the stochasticity of the environment.
The idea is that r(.; θ, E) implements f (.; θ) in E. This means that r(.; θ, E) computes the same function as f (.; θ), and the (theoretically) associative operations are executed in some (potentially random) order defined by E, and the number representation and rounding details are also defined by E.
Let us now elaborate on r(.; θ, E) and discuss a few obser-vations about the relationship of f (.; θ) and r(.; θ, E).
this section cite: []

Section: Domain and range.
The domain X is restricted, more precisely, X ⊂ R n , where X is defined by the representable numbers in R n within E. The range Y behaves similarly (assuming E is deterministic), that is, Y ⊂ R m , where Y is defined by the representable numbers in R m within E.
Range in stochastic environments. The environment can be stochastic, depending on a number of factors such as randomly changing operation order, or even unpredictable rounding due to using several different platforms in parallel.
In this case, we do not model the distribution of the output. Instead, we define the range to be the representable subsets of R m , and the value of r(x; θ, E) is the subset containing the vectors with a probability larger than zero. In other words, Y ⊂ 2 R m , defined by the representable numbers in 2 R m within E.
this section cite: []

Section: Different Output.
Even for a representable input x ∈ X, and a deterministic E, we have r(x; θ, E) ̸ = f (x; θ) in general. First of all, θ might not be representable exactly. Even if θ is representable, due to floating point issues, the output of the deployed network will typically be different, except in rare cases when all the sub-computations are exactly representable. In Section 7 we argue that these differences can be arbitrary.
this section cite: []

Section: Verification of Deployed Networks
Given an input x * ∈ X, the verification problem is now to prove that
∀x ∈ D ϵ,p,E (x * ) ∀z ∈ r(x; θ, E), z ≥ 0,(2)
where D p,ϵ,E is defined below. The problem is formulated assuming a stochastic environment, but note that the deterministic environment is a special case of the stochastic one. Apart from this, and the differences between f and r discussed above, there are other significant differences from Equation (1) regarding D ϵ,p,E (x * ) and the underlying property P E (x * ).
this section cite: []

Section: Verified domain.
For an x * ∈ X, the verification problem will have a domain
D p,ϵ,E = {x ∈ X : ∥x -x * ∥ p ≤ ϵ}. Now, it is unclear whether D p,ϵ,E = D p,ϵ ∩ X
because the rounding issues affecting D p,ϵ,E might depend on the verification algorithm as well, but for simplicity we will use D p,ϵ,E = D p,ϵ ∩ X as a definition of D p,ϵ,E . Verified property. The properties P and P E significantly differ because we have P E (x * ) = {x ∈ X : r(x; θ, E) y(x * ) = max i r(x; θ, E) i }. Thus, in general, P E ̸ ⊂ P because some negative outputs of f might become zero for r due to rounding. So, here, we do not have P E = P ∩ X. (Also, obviously, P ̸ ⊂ P E .)
this section cite: []

Section: No Soundness in Deployment
Clearly, practically complete (but not sound) solutions are known: every solution based on a heuristic search for adversarial examples in the given domain is a complete method for the deployed problem, e.g., (Kurakin et al., 2017;Madry et al., 2018), although only if the search uses the deployed implementation of the network. We only need the method to return true if no adversarial input is found: if there are no adversarial inputs, none will be found, making the method complete.
Thus, we focus on soundness. Here, we examine theoretically sound methods from the literature (see Table 3 for a list) and investigate whether they are practically sound, that is, whether they are sound regarding the deployed verification problem in Equation ( 2). We will show that in general the answer is no. In particular, we are not aware of any method that is sound in stochastic environments, despite explicit claims (Singh et al., 2025).
In the following, we first discuss preliminary assumptions and then we address techniques applied in the literature to achieve soundness, indicated in Table 3: interval bound propagation (IBP) (Gowal et al., 2018) that is based on interval arithmetic (Alefeld & Herzberger, 1983), and more advanced propagation methods based on symbolic approaches such as affine arithmetic (de Figueiredo & Stolfi, 2004).
this section cite: ['b14', 'b18', 'b29', 'b9', 'b1']

Section: Preliminaries
In our theoretical discussion, we shall focus on a very simple class of functions: the sum of the input variables. The input variables will be zero-length intervals, that is, constants that are representable in the deployment environment. Since the problems we identify can be demonstrated already in this simple function class with constant inputs, generalizations of this class, such as neural network architectures, will inherit these problems.
The sum computation is associative, furthermore, it can be parenthesized arbitrarily, leading to a large variety of possible binary expression trees. Here, we will assume that the environment E defines two aspects: the floating point representation (along with the rounding mode) and the set of expression trees with non-zero probability. Thus, here, an expression tree defines a deterministic hierarchy of binary sum operations, all of which are computed using the same precision and rounding mode, leading to a deterministic output for any input given in the leaves.
Since the sum function does not use any parameters, we will omit θ from our notation. Thus, here, f (x) = i x i and r(x; E) is the deployed version of the sum using the number representation and expression tree set defined by E.
Let us introduce the notation r(x; E) = {r(x; E, o i ) : o i is a possible expression tree in E}, and let L r and U r be the minimum and maximum elements of r(x; E), respectively.
The proofs of the propositions are given in Appendix A.
this section cite: []

Section: Interval Bound Propagation
When using interval arithmetic on a given expression tree, each binary addition will have two intervals as inputs and will output an interval that also considers the numerical error due to rounding: if the lower or upper bound is not representable, it is rounded towards -∞ or ∞, respectively.
Our first proposition will state a nice soundness property of IBP in two particular cases: (1) when we wish to bound the full-precision value, and (2) when we wish to bound the floating point value and we also know a specific expression tree that minimizes or maximizes this value. Proposition 6.1. Let [a, b] o = f ([x, x]) (x ∈ X) be the interval evaluation of f using expression tree o and the floating point representation defined by E. Then, ( 1
) f (x) ∈ [a, b] o , (2) if r(x; E, o) minimizes or maximizes r(x; E) then a ≤ L r or U r ≤ b, respectively.
In the special case of a deterministic environment, this means that if we know the deterministic expression tree and compute IBP accordingly, then IBP will be sound, since this expression tree will both minimize and maximize the floating point output. Unfortunately, if we do not know the deterministic order, or if the environment is stochastic, then there is no guarantee that IBP will be practically sound, as formulated by the next proposition. Proposition 6.2. For any environment E that allows for every correct expression tree and uses IEEE 754 floating point representation with any fixed rounding mode, there is an expression tree o, and input x ∈ X, such that for the interval evaluation [a, b]
o = f ([x, x]) in environment E we have L r < a or b < U r .
Recall, that in practice, deployment environments are often inherently stochastic (Schlögl et al., 2023). Proposition 6.2 means that in stochastic environments we must find an expression tree that guarantees practical soundness, because not all of them do in general. This task depends on the environment but, for example, the expression tree that minimizes the output of r(.; E) is a suitable choice for bounding the output from below in any environment according to Proposition 6.1.
this section cite: ['b23']

Section: Notes on Computational Complexity
Practically sound verification with IBP, thus, requires finding special expression trees. One might think that some orderings-such as adding numbers in a decreasing order, or maybe in a decreasing order according to absolute valuemight minimize r(.; E) and thus might be suitable to make IBP sound. This is not the case.
Proposition 6.3. For any environment E that allows for every correct expression tree and uses IEEE 754 floating point representation with any fixed rounding mode, there is an input x ∈ X, such that we have L r < r(x; E, o) for o ∈ {decreasing-order, decreasing-absolute-value-order}.
Note that good approximation algorithms might exist for the minimum and maximum output, but in a verification mindset, we wish to eliminate any mistakes completely, otherwise the system remains vulnerable to attacks.
We hypothesize that finding the expression tree that maximizes or minimizes the output is an NP-hard problem in general. It was shown in (Kao & Wang, 2000) that a very similar problem, namely finding the order that minimizes the worst-case error is NP-hard in the general case when both positive and negative numbers are added.
this section cite: ['b12']

Section: Symbolic Bound Propagation
Due to the so called dependency problem, naive interval arithmetic might greatly overestimate the output interval of multivariate functions of a large complexity. Symbolic approaches mitigate this problem via propagating symbolic representations instead of values. We briefly summarize two important approaches here, for more details and proofs please refer to Appendix A.1.
Our main observation is that, in our special case of computing the sum of zero-length intervals, these symbolic methods simplify to non-symbolic interval methods.
Polyhedra-based approaches. In this special case, polyhedra-based approaches like DeepPoly (Singh et al., 2019a) and CROWN (Zhang et al., 2018;Xu et al., 2021;Wang et al., 2021) first find the symbolic expression for the sum of the input variables. During sound verification, however, the implementations evaluate this formula using interval arithmetic, based on some binary expression tree defined by the verifier. Thus, the method is equivalent to IBP discussed in Section 6.2. In other words, this method is not practically sound either in the general case.
Zonotope-based approaches. DeepZ (Singh et al., 2018) and RefineZono (Singh et al., 2019b) are based on the zonotope abstraction, using affine arithmetics (de Figueiredo & Stolfi, 2004) to propagate bounds throughout the network, computing the so-called zonotope domain (Ghorbal et al., 2009). This case, too, simplifies to interval analysis in our special case, only the intervals will be strictly wider than those computed by IBP due to the application of a technique proposed in (Miné, 2004).
Since this latter case is not equivalent to IBP, we explicitly state propositions describing the zonotope domain, indicating that the verification is not practically sound in general, very similarly to IBP.
Proposition 6.4. Proposition 6.1 holds also when computing the interval using the widening technique in (Miné, 2004) used by (Singh et al., 2019b).
Proposition 6.5. Proposition 6.2 holds also when computing the interval using the widening technique in (Miné, 2004) used by (Singh et al., 2019b).
this section cite: ['b38', 'b37', 'b34', 'b26', 'b4', 'b8', 'b19', 'b19', 'b19']

Section: Practical Attacks on Verifiers
Here, based on our the theoretical results, we design neural networks with the purpose of fooling state-of-the-art sound verifiers in practical deployment. The basic idea is that we design small detector neurons that are triggered by certain properties of an environment-such as precision or the order of computation-in order to activate arbitrary adversarial behavior.
Adversarial networks. These detector neurons can be inserted into any neural network using the method proposed in (Zombori et al., 2021). This way, they can enable backdoors that can alter the behavior of the network arbitrarily, triggered by a specific property of the environment. In Appendix B we discuss the technical details of creating the complete backdoored networks. Here, we detail our detector neurons.
The detector neuron. The generic scheme of a detector neuron is the following. We use a linear neuron with n inputs without an activation function. We assume that the neuron gets the constant input vector of all ones (1, . . . , 1). This way, the output of the neuron is w 1 + • • • + w n + b, that is, we sum the edge weights and the bias. We will call this order of summands without parenthesis the default expression tree, often implemented in environments using a single-threaded CPU.
Note that assuming a constant input for the detector neurons is not essential because, on the one hand, our constructions can easily be generalized to a random input point from a bounded interval, and on the other hand, even a constant input can be fabricated when embedding the neuron into the host network (see Appendix B).
Preliminaries. We assume that the deployment environment uses IEEE 754 floating point representation with the default round-to-nearest rounding. We will use a special number ω, which is defined as the smallest representable positive number such that the next representable number after ω is ω + 2. For example, for the binary32 format ω = 2 24 , and for the binary64 format ω = 2 53 . Using the default rounding mode, ω + 1 = ω.
this section cite: ['b40']

Section: Detecting Precision
We define the detector neuron as ω + 1 -ω. Assuming the default expression tree (summing left to right) this detector neuron will output 0 if the precision of the deployment is the same as the precision represented by ω, and it will return 1 in higher precision deployments.
For a verifier that uses a given precision during verification, we will integrate this detector neuron into the adversarial network in such a way that the adversarial behavior will be triggered by a precision different from the one used by the verifier. This way, we can test whether the verifier is able to cover every precision that is possible in deployment.
this section cite: []

Section: Detecting Expression Trees
For detecting expression trees, we define three different detector neurons. The idea behind these definitions is that the three detectors represent an increasingly challenging problem for verifiers.
In each case, these detectors will be integrated into adversarial networks so that using the default expression tree activates the normal behavior of the network. To be more precise, if evaluating the detector in the default order results in zero then the adversarial behavior is triggered by any non-zero output of the detector, and if the default order results in a non-zero output then the output of zero triggers the adversarial behavior.
this section cite: []

Section: A VERIFIER-FRIENDLY DETECTOR
The detector neuron returns the following sum of (2h 1 + 1)h 2 summands:
ω h 1 + • • • + ω h 1 h1× +1 + -ω h 1 + • • • + -ω h 1 h1× + • • • h2× ,(3)
where the bias weight is zero and thus omitted.
Assuming the default expression tree the value of this sum is 0 in deployment, thus, the value 0 will trigger normal behavior.
For verifiers, dealing with this sum is relatively easy, because even simple interval arithmetic will be sound, covering every possible order, if the analysis is executed according to the default expression tree. At the same time, this trigger will be activated often, because there is a large number of possible orderings that return a value from [1, h 2 ].
Note that the full-precision value is h 2 . In our experiments we will use h 1 = 4 and h 2 = 15.
this section cite: []

Section: VERIFIER-UNFRIENDLY DETECTORS
Our first unfriendly detector returns the sum of h + 2 summands
2 h + • • • + 2 h h× +ω -ω(4)
Assuming the default expression tree the value of this sum is 2 in deployment, thus, any non-zero value will trigger the normal behavior, and the value of zero triggers adversarial behavior.
It is a harder expression for verifiers, because now, a verification using interval arithmetic on the default expression tree will no longer be sound: it will not contain the output 0, which is a possible output. Indeed, if we sum the edge weights in an order where ω is in a position earlier than position h/2 and we add the bias -ω in the end, the result is 0. But, in this case, there is a large number of orders that do result in covering 0 (and thus a practically sound behavior). In fact, the only order of the edge weights that does not result in covering 0 is the default order.
For verification, the hardest problem we will test is the sum
1 + • • • + 1 h× +ω -ω,(5)
which has much fewer possible orderings of the edge weights that result in covering the output 0 using interval arithmetic. In fact, we cover zero only if ω is first or second in the order of summation (assuming the bias is added last). This means that verifiers that are not guaranteed to cover all possible orders will almost certainly fail to detect the adversarial behavior.
As for the parameter h, we set h = 512 for both unfriendly detectors.
this section cite: []

Section: Empirical Evaluation
To demonstrate the practical relevance of our observations, we evaluated state-of-the-art verifiers using our adversarial networks to test whether they indeed cover all possible executions in a number of different deployment environments.
As predicted, the answer we found is no.
this section cite: []

Section: Adversarial Networks
In our evaluation, we used an MNIST network checkpoint made available by (Wong & Kolter, 2018). This checkpoint was used to evaluate MIPVerify (Tjeng et al., 2017) as well as the attacks proposed by (Zombori et al., 2021). This network has two convolutional layers with stride 2: one with 16 and one with 32 filters of size 4×4, followed by a 100 neuron fully connected layer. ReLU activations are used by all the neurons. The network was trained to be robust against attacks within a radius of 0.1 in the l ∞ -norm (Wong & Kolter, 2018).
We augmented this network with our detectors presented in Section 7 using the methodology proposed in (Zombori et al., 2021), as detailed in Appendix B. We created five adversarial networks: two precision-based attacks (one with adversarial behavior in 32-bit environments, and one in 64bit environments) based on the detector in Section 7.1, and three expression tree based attacks described in Section 7.2 based on the detectors in Equations ( 3) to ( 5). We will refer to these three attacks as Order1, Order2, and Order3, respectively.
this section cite: ['b35', 'b31', 'b40', 'b35', 'b40']

Section: Effectiveness of our Adversarial Networks
Here, we wish to demonstrate that the adversarial behavior of our networks is indeed activated in certain environments.
this section cite: []

Section: PRECISION ATTACKS
In Table 1, the accuracy of the adversarial networks with precision attacks is presented. The first row corresponds to the configuration where the backdoor changes the prediction under 32-bit precision, while the second row corresponds the 64-bit precision trigger. The columns correspond to 32-bit and 64-bit inference. It is clearly visible that our backdoors indeed trigger adversarial behavior as designed. Note that the adversarial behavior has larger than zero accuracy due to some of the test examples that are misclassified by the host model.
this section cite: []

Section: EXPRESSION TREE ATTACKS
The environments. The effective expression tree in deployment depends on numerous factors, such as hardware specifications, memory size, underlying linear algebra libraries, machine learning frameworks, optimizations, batch size during inference, and so on (Schlögl et al., 2023). We defined 8 different environments that provide a sufficient diversity to support our claims. The libraries we tested were PyTorch and Flux on Julia. As for hardware, we tested CPU and GPU, and we also tested different batch sizes during inference. For a more detailed description please refer to Appendix C.1.
Results. Table 2 contains the accuracy results over different environments. Three of our backdoors are activated in at least one environment, which gives further support that these environments execute different expression trees, including those that we used as backdoors. The Order3 adversarial network is an exception because-as we explained beforeit has very few expression trees that trigger its adversarial behavior. Nevertheless, this does not make the Order3 network safe, and this fact should still be detected by sound verifiers.
this section cite: ['b23']

Section: Pool of Verifiers
Table 3 lists the verifiers that we tested. Here, we briefly describe them, for more detail, please refer to Appendix C.2.
MIPVerify, RefineZono, and RefinePoly are claimed to be sound and complete verifiers. They rely on a cheap sound bounding method that is then refined using mixed-integer linear programming (MILP) to achieve completeness.
RefineZono and RefinePoly rely on DeepZ and DeepPoly as their sound bounding method, respectively. DeepZ and DeepPoly are both claimed to be sound under floating-point arithmetic.
β-CROWN BaB is built on the fast and sound (but incomplete) bound propagation algorithm β-CROWN and integrates it with a branch-and-bound framework to ensure completeness. The verification process can be configured to run on either CPU or GPU and supports numerical representations in both 32-bit and 64-bit precision. GCP-Crown is also a BaB method similar to β-CROWN, but it is capable of managing general cutting-plane constraints.
this section cite: []

Section: Attacking Verifiers
Here, we verify our adversarial networks with our pool of verifiers. Note that here, the verifiers do not consider the deployment environments explicitly, instead, the verifiers themselves have their own deployment environments, in which they are executed as indicated in Table 3 (see Appendix C.2 for a detailed discussion).
As for the deployment environments of our networks, we know from Section 8.2.2 that for all the adversarial networks there is at least one environment where the adversarial behavior is observed, except for Order3. Thus, it is a rightful expectation that the verifiers find the planted backdoor.
In the case of Order3, we argue that we should also require the verifiers to find the backdoor, because in general it is very hard to explicitly model the set of possible expression trees in a given complex environment, so practically sound verifiers should be prepared for all of them.
Table 3. The vulnerability of the verifiers to our four attacks and to the attack of (Zombori et al., 2021). The Precision attack is adversarial to the 32-bit environment, except for 32-bit β-CROWN, where it is adversarial to 64-bit.
Verifier Ver. Env.
this section cite: ['b40']

Section: Bounding Precision Order1 Order2
Order3 (Zombori et al., 2021) MIPVerify (Tjeng et al., 2017) 64-bit, CPU IBP unsound sound unsound unsound unsound MN-BAB (Ferrari et al., 2022) 64-bit, GPU Polyhedra unsound sound unsound unsound unsound β-CROWN BaB (Wang et al., 2021) 32-bit, CPU Polyhedra unsound sound unsound unsound [no 32-bit model] β-CROWN BaB (Wang et al., 2021) 64-bit, CPU Polyhedra unsound sound unsound unsound sound β-CROWN BaB (Wang et al., 2021) 64-bit, GPU Polyhedra unsound sound unsound unsound sound GCP-CROWN (Zhang et al., 2022) 64-bit, CPU Polyhedra unsound sound unsound unsound sound DeepPoly (Singh et al., 2019a) 64-bit, CPU Polyhedra unsound sound sound unsound sound RefinePoly (Singh et al., 2019a) 64-bit, CPU Polyhedra unsound sound sound unsound unsound DeepZ (Singh et al., 2018) 64-bit, CPU Zonotope unsound sound sound unsound sound RefineZono (Singh et al., 2019b) 64-bit, CPU Zonotope unsound sound sound unsound [Gurobi error]
Table 3 contains the results of the verification of the first 100 examples from the MNIST test set. The label sound indicates that the verifier was able to detect the backdoor each time, returning a 0% verified robust accuracy. Otherwise the result is labeled unsound.
this section cite: ['b40', 'b31', 'b7', 'b34', 'b34', 'b34', 'b39', 'b26']

Section: Discussion
No practically sound verifiers. The results in Table 3 reveal that none of the listed approaches are practically sound, given that they missed the backdoor in the most difficult Order3 network as well as in the Precision network.
this section cite: []

Section: References
Ref_id:b0 Title: Introduction to neural network verification Year: (2021)
Ref_id:b1 Title: Introduction to Interval Computation Year: (1983)
Ref_id:b2 Title: Branch and bound for piecewise linear neural network verification Year: (2020)
Ref_id:b3 Title: Neural network verification is a programming language challenge Year: (2025)
Ref_id:b4 Title: Affine arithmetic: Concepts and applications Year: (2004)
Ref_id:b5 Title: Output range analysis for deep feedforward neural networks Year: (2018)
Ref_id:b6 Title: Formal verification of piece-wise linear feedforward neural networks Year: (2017-10-03)
Ref_id:b7 Title: Complete verification via multi-neuron relaxation guided branch-and-bound Year: (2022)
Ref_id:b8 Title: The zonotope abstract domain taylor1+ Year: (2009)
Ref_id:b9 Title: On the effectiveness of interval bound propagation for training verifiably robust models Year: (2018)
Ref_id:b10 Title: A survey of safety and trustworthiness of deep neural networks: Verification, testing, adversarial attack and defence, and interpretability Year: (2020)
Ref_id:b11 Title: Exploiting verified neural networks via floating point numerical error Year: (2021)
Ref_id:b12 Title: Linear-time approximation algorithms for computing numerical summation with provably small errors Year: (2000-03)
Ref_id:b13 Title: An efficient smt solver for verifying deep neural networks Year: (2017)
Ref_id:b14 Title: Adversarial machine learning at scale Year: (2017-04-24)
Ref_id:b15 Title: Certified robustness for deep neural networks Year: (2023)
Ref_id:b16 Title: Algorithms for verifying deep neural networks Year: (2021)
Ref_id:b17 Title: Sound mixed fixed-point quantization of neural networks Year: ()
Ref_id:b18 Title: Towards deep learning models resistant to adversarial attacks Year: (2018)
Ref_id:b19 Title: Relational abstract domains for the detection of floating-point run-time errors Year: (2004)
Ref_id:b20 Title: Differentiable abstract interpretation for provably robust neural networks Year: (2018-07)
Ref_id:b21 Title: Scaling the convex barrier with active sets Year: (2021)
Ref_id:b22 Title:  Year: (2014)
Ref_id:b23 Title: Causes and effects of unanticipated numerical deviations in neural network inference frameworks Year: (2023)
Ref_id:b24 Title: Impacts of floatingpoint non-associativity on reproducibility for HPC and deep learning applications Year: (2024)
Ref_id:b25 Title: Robustness of deep learning classification to adversarial input on GPUs: asynchronous parallel accumulation is a source of vulnerability Year: ()
Ref_id:b26 Title: Fast and effective robustness certification Year: (2018-12-03)
Ref_id:b27 Title: An abstract domain for certifying neural networks Year: (2019-01)
Ref_id:b28 Title: Boosting robustness certification of neural networks Year: (2019)
Ref_id:b29 Title: ERAN user manual Year: (2025)
Ref_id:b30 Title: Intriguing properties of neural networks Year: (2014)
Ref_id:b31 Title: Evaluating robustness of neural networks with mixed integer programming Year: (2017)
Ref_id:b32 Title: Effects of floatingpoint non-associativity on numerical computations on massively multithreaded systems Year: (2009)
Ref_id:b33 Title:  Year: ()
Ref_id:b34 Title: Beta-crown: Efficient bound propagation with per-neuron split constraints for neural network robustness verification Year: (2021-12-06)
Ref_id:b35 Title: Provable defenses against adversarial examples via the convex outer adversarial polytope Year: (2018-07)
Ref_id:b36 Title: Automatic perturbation analysis for scalable certified robustness and beyond Year: (2020)
Ref_id:b37 Title: Fast and Complete: Enabling complete neural network verification with rapid and massively parallel incomplete verifiers Year: (2021)
Ref_id:b38 Title: Efficient neural network robustness certification with general activation functions Year: (2018-12-03)
Ref_id:b39 Title: General cutting planes for boundpropagation-based neural network verification Year: (2022-12-09)
Ref_id:b40 Title: Fooling a complete neural network verifier Year: (2021)
