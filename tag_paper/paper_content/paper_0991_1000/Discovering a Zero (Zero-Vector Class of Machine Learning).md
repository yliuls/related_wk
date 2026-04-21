Title: Discovering a Zero (Zero-Vector Class of Machine Learning)
Abstract: In Machine learning, separating data into classes is a very fundamental problem. A mathematical framework around the classes is presented in this work to deepen the understanding of classes. The classes are defined as vectors in a Vector Space, where addition corresponds to the union of classes, and scalar multiplication resembles set complement of classes. The Zero-Vector in the vector space corresponds to a class referred to as the Metta-Class. This discovery enables numerous applications. One such application, termed 'clear learning' in this work, focuses on learning the true nature (manifold) of the data instead of merely learning a boundary sufficient for classification. Another application, called 'unary class learning', involves learning a single class in isolation rather than learning by comparing two or more classes. Additionally, 'set operations on classes' is another application highlighted in this work. Furthermore, Continual Learning of classes is facilitated by smaller networks. The Metta-Class enables neural networks to learn only the data manifold; therefore, it can also be used for generation of new data. Results for the key applications are shown using the MNIST dataset. To further strengthen the claims, some results are also produced using the CIFAR-10 and ImageNet-1k embeddings. The code supporting these applications is publicly available at: github.com/hm-4/Metta-Class.

Section: Introduction
There are many techniques in Machine learning that allow data classification, one such contraption is the Neural Network. A general technique used in the data classification is the following. Find a separating hyper-surface between classes such that functional value of data points belong to same class when substituted in the hyper-surface equations resulting in a 'similar' sets of values. In Neural Networks those set of values are called logits. The general technique, with respect to Neural Networks, can be understood by thinking the logit equations of a Neural Network as hypersurface equations, those logit values are passed though the Sigmoid or Softmax layer. The similarity of the logits of data points belonging to the same class is enforced by the cross entropy loss.
An example of decision boundaries of a neural network is shown in the far-left subfigure of Figure 1. The feature space in that subfigure is divided into two decision regions: the lower half is classified as Class-2, indicated by the green region, while the upper half is classified as Class-1, represented by the red region. However, this dichotomization of the entire feature space into these two classes is not accurate, as only a small portion of the space genuinely belongs to Class-1 or Class-2. Therefore, the decision regions depicted in the far-left subfigure of Figure 1 do not accurately represent the true nature of the classes. A more realistic representation of the decision regions can be seen in the mid-subfigure of Figure 1.
If a neural network exhibits the decision regions as shown in the mid-subfigure of Figure 1, these regions have particularly notable set properties. One such property is the union of the regions: the union of the decision regions accurately represents the true nature of the combined class's decision region. Another property is the complement of the decision region: as illustrated in the far-right subfigure of Figure 1, the complement of a decision region accurately represents the true complement of the class. These properties are essential, as they reflect the true characteristics of the classes. Consequently, neural networks with such well-defined decision regions for each class are desirable.
In this work, classes are characterized in terms of the logit equations, and logit equations are viewed as vectors. Addition and scalar multiplication on such vectors is defined such that a vector space is created. As a consequence, it allows to use all vector operations on logits hence enables set operations on classes. The relevant proofs and meanings of the operations are provided, without going into very rigorous math. Sample PyTorch implementations and results are attached supporting the theory when needed.
this section cite: []

Section: Characterization of Logits
If the probability density function (PDF) of the data, f n (x), is known, a given data point can be assessed for membership in the distribution by comparing its probability to a predefined threshold, α fn . The threshold α fn should be chosen as a non-negative real number, ensuring that data points with high relative probability are recognized as belonging to the class. As illustrated in Figure 2, data points with probabilities (f n (x)) greater than or equal to α fn = 0.1 are classified as belonging to Class-f , while those with lower probabilities are classified as not belonging to the class. The PDF function f n (x) itself is used for this classification along with the threshold α fn .
In cases where only the unnormalized PDF f (x) is known, it is still possible to classify data points by comparing f (x) to the scaled threshold α f A f , where A f represents the normalization constant of f (x). A diagram illustrating this mechanism can be seen in Figure 2, where A f = 1.
The value f (x 0 ) for a given data point x 0 is referred to as the 'Logit.' The function f (x) is defined as the 'Logit Function' or 'Logit Equation.' Any Logit Equation that correctly classifies data with respect to a predefined threshold α f is termed a 'Valid Logit Equation (VLE)' or 'Valid Logit Function.' These definitions will be used consistently throughout this paper.
All Valid Logit Functions can be grouped into a set, as shown in equation 1. Any function h(x) that belongs to the set [f (x)] in equation 1 can classify data sampled from the unnormalized distribution f (x). Notice that functions of the form h(x) + k, where h(x) ∈ [f (x)] and k ∈ R, with a threshold α f + k can perform the same classification task as h(x) and α f . Therefore, to accommodate equations of the form h(x) + k and thresholds of the form α f + k, as well as any other valid pair of functions and thresholds, a separate set S is introduced in equation 1.
[f (x)] = β f (x) -
α f A f + α f β ∈ R + ∪ S (1
)
Here, f (x) represents the unnormalized PDF of the class, α f ∈ R is the predefined threshold for f (x), A f represents the normalization constant of f (x), and S denotes the set of unknown (does not need to be an unnormalized or a normalized PDF) Valid Logit Functions associated with that class.
The set [f (x)] in equation 1 is an equivalence class with respect to the classification of the data, as it satisfies the properties of an equivalence relation. Specifically, any equation within this set classifies the data in exactly the same way as others in the set for a predefined threshold. Consequently, the set [f (x)] is referred to as the 'Valid Equivalence Set' of f (x). This terminology will be consistently used throughout this paper.
The following subsection explores the resulting Valid Equivalence Set when two classes are combined.
this section cite: []

Section: Merging two Classes
Now consider another class, Class-2, whose unnormalized PDF is g(x). Figure 3 illustrates some Valid Logit Functions for Class-g, similar to those shown for Class-f in Figure 2. Any function k(x) belonging to the set [g(x)], as defined in Equation ( 2), is a Valid Logit Function capable of classifying data sampled from the unnormalized distribution g(x), in the same way as was previously done for Class-f .
[g(x)] = β g(x) -α g A g + α g β ∈ R + ∪ T (2) -6 -4 -2 0 2 4 x -0.1 0.0 0.1 0.2 0.3 0.4 0.5 αg =0.05 g(x)
2(g(x) -0.05) + 0.05 0.5(g(x) -0.05) + 0.05 20(g(x) -0.05) + 0.05
this section cite: []

Section: Threshold Line
Figure 3. A set of functions that classify Class-g (g(x)) with the exact same decision regions for a threshold of αg = 0.05.
-6 -4 -2 0 2 4
x -0.1 Here, g(x) represents the unnormalized PDF of Class-g, α g ∈ R is the predefined threshold for g(x), A g represents the normalization constant of g(x), and T denotes the set of unknown (does not need to be an unnormalized or a normalized PDF) Valid Logit Functions associated with Class-g. Suppose there is a need to merge the data of Class-g with the data of Class-f . The resulting PDF of the combined class, denoted as u(x), is illustrated in Figure 4. Assuming an equal number of data points in Class-f and Class-g, the equation for u(x) in terms of PDFs f n (x) and g n (x), the normalized distributions of f (x) and g(x) respectively, is given in Equation ( 3). The proof of this equation is provided in Section A of the Appendix.
0.0 0.1 0.2 0.3 0.4 0.5 αu =0.12 u(x) = f (x)+g(x)
u(x) = f n (x) + g n (x) 2 (3
)
Similar to what was observed for Class-f and Class-g, there can be multiple Valid Logit Functions corresponding to the combined PDF u(x), as illustrated in Figure 4. The set [u(x)], as defined in Equation ( 4), represents all Valid Logit Functions associated with the PDF u(x).
[u(x)] = β u(x)α u A u + α u β ∈ R + ∪ Z (4)
-6 -4 -2 0 2 4 x -0.1 0.0 0.1 0.2 0.3 0.4 0.5 α-f =0.15 f (x) 2(-f (x) + 0.15) + 0.15 0.5(-f (x) + 0.15) + 0.15 20(-f (x) + 0.15) + 0.15 (-f (x) + 0.15) + 0.15 Threshold Line Figure 5. A set of functions that classify complement of Class-f with the exact same decision regions for a threshold of α -f = 0.15.
Here, u(x) represents the PDF of the combined distribution, α u ∈ R is a predefined threshold for PDF u(x), A u represents the normalization constant of u(x), and Z denotes the set of unknown (does not need to be an unnormalized or a normalized PDF) Valid Logit Functions associated with the combined distribution u( x).
An important question arises: given the logit equations h(x) ∈ [f (x)] and k(x) ∈ [g(x)], is it possible to construct a new logit equation v(x) ∈ [u(x)] for the combined distribution? If achievable, v(x) could be utilized for classification of the combined distribution with PDF u(x). However, this is not feasible, as numerous counterexamples can be identified. A related question then follows: can the sets [f (x)] and [g(x)] themselves be combined to derive a new set [u(x)], as shown in Equation ( 5).
[f (x)] ∪ [g(x)] =[u(x)](5)
[f (x)] + [g(x)] = f n (x) + g n (x) 2 (6) [f (x)] + [g(x)] =[f n (x) + g n (x)](7)
Notice that the sets [f n (x) + g n (x)] and fn(x)+gn(x) 2 are exactly the same. The '+' operator in the left-hand side of Equation ( 6) is used as a symbol to represent union. However, the '+' operator on the right-hand side of the same equation denotes the standard addition operator. This choice is intentional, as the equation is later adopted as a definition of addition in the following section.
this section cite: []

Section: Complement of a Class
There is an observation on the Valid Equivalence Set [-f (x)]: it represents the complement of Class-f , as illustrated in Figure 5 for the same predefined threshold α f . Mathematically, this is expressed as in Equation ( 8).
[f (x)] c =[-f (x)] (8) -[f (x)] =[-f (x)](9)
The '-' operator on the left-hand side of Equation ( 9) is used as a symbol to represent the complement operation. However, the '-' operator on the right-hand side of the same equation denote standard subtraction operation. This choice is intentional, as the equation is later adopted as a definition in the following section.
this section cite: []

Section: The Vector Space
As seen in the previous chapter, the set of Valid Logit Equations of a class, whose PDF is f (x), is called the Equivalence set and represented by the [f (x)]. Therefore the function space is divided into set of these equivalence sets each representing a class.
In this chapter, all the properties of a vector space (Axler, 2015) are verified for that set of equivalence sets, using the definitions of addition 3.2.1 and scalar multiplication 3.2.2. During this process a zero is discovered for this vector space, this zero represents an equivalence set. As previously seen each such equivalence set represents a data class. The data class represented by the zero element of the vector space is referred to as the Metta-Class (also called the Zero-Vector Class).
this section cite: ['b0']

Section: Definition of set V
Let V be the set of equivalence sets. Each equivalence set also known as Valid Logit Function set represents a data class. If the PDF of a data-class is f (x) then the corresponding equivalence set is denoted by [f (x)]. Therefore the set V can be written as
V = { [f (x)], [g(x)], [h(x)], .... }(10)
Throughout this report, whenever the vector space V is mentioned, it refers to this vector space.
this section cite: []

Section: Definition of Addition and Scalar multiplication
As understood in the previous chapter, the equations for union (7) and complement (9) are taken as definitions here with slight modifications so that they are suitable for defining a vector space. Specifically,
this section cite: []

Section: ADDITION
For all [f (x)], [g(x)] ∈ V , the sum [f (x)] + [g(x)] is defined as
[f (x)] + [g(x)] := [f (x) + g(x)](11)
this section cite: []

Section: SCALAR MULTIPLICATION
For all λ ∈ R and for all [f (x)] ∈ V the product λ[f (x)] is defined as λ [f (x)] := [λ f (x)]
3.3. Verification of the Properties of vector space on set V, along with addition and scalar multiplication
this section cite: []

Section: COMMUTATIVITY
For all [f (x)], [g(x)] ∈ V using the property of addition
[f (x)] + [g(x)] = [f (x) + g(x)] = [g(x)] + [f (x)]
Hence the commutativity property of vector space holds on set V.
this section cite: []

Section: ASSOCIATIVITY
For all [f (x)], [g(x)], [h(x)] ∈ V , using the property of addition
[f (x)] + [g(x)] + [h(x)] = [f (x) + g(x)] + [h(x)] = [f (x)] + [g(x) + h(x)]
let a, b ∈ R, then using the property of scalar multiplication
(ab)[f (x)] = a b[f (x)] = a [bf (x)]
Hence the associativity property of vector space holds on set V.
this section cite: []

Section: ADDITIVE IDENTITY (THE ZERO)
For all [f (x)] ∈ V there exists 0 such that
[f (x)] + [0] = [f (x) + 0] = [f (x)]
Hence there exists an additive identity on set V.
this section cite: []

Section: ADDITIVE INVERSE
For every [f (x)] ∈ V there exists [-f (x)] such that
[f (x)] + [-f (x)] = [f (x) + -f (x)] = [0]
Hence there exists additive inverse on set V.
this section cite: []

Section: MULTIPLICATIVE IDENTITY
For every [f (x)] ∈ V , 1[f (x)] = [f (x)], hence the multiplicative identity exists.
3.3.6. DISTRIBUTIVE PROPERTIES For all a, b ∈ R and for all [f (x)], [g(x)] ∈ V a([f (x)] + [g(x)]) = a[f (x)] + a[g(x)] = [af (x)] + [ag(x)] (a + b)[f (x)] = (a[f (x)] + b[f (x)]) = [af (x)] + [bf (x)]
Hence the distributive properties of vector space holds on set V.
Therefore, the defined set V forms a vector space under the specified operations of addition and scalar multiplication.
The equivalence sets are represented as vectors in this space and are referred to as class-vectors, with each vector corresponding to a distinct data class. In particular, the zero vector ([0]) in this space might also represents a data class, referred to as the Zero-Vector Class.
To avoid ambiguity between the concept of the Zero-Vector Class and the label "zero" used for other classes, this class is hereafter referred to as the Metta-Class.
this section cite: []

Section: Metta-Class and its Applications
As established in the previous chapter, the set V defined in Section 3.1, together with the addition and scalar multiplication operations described in Subsections 3.2.1 and 3.2.2, forms a vector space. Each vector in the vector space V represents a data class. The additive identity vector [0] of this space (see Subsection 3.3.3) must therefore also represent a data class. In the previous chapter, the class corresponding to this additive identity vector was named the Metta-Class.
Using the equation 1, the set [0] can be written as
[0] = β 0 -α 0 A 0 + α 0 β ∈ R +(13)
Here, α 0 is a threshold for PDF of the class if we define normalizing constant A f of unnormalized PDF f (x) as
A f := lim V→[-∞,∞] d V f (x) dx (14
) Simple substitution f (x) = 0 in Equation (14) gives A 0 = 0 hence [0] = {α 0 } (15
)
where α 0 ∈ R. Therefore, the logit equation of the Metta-Class is a constant valued function.
If f (x) = 0 is treated as a constant going to zero in the limit i.e., f (x) = lim k→0 k then equation 14 becomes
A 0 = lim V→[-∞,∞] d V lim k→0 k dx (16
) = lim V→[-∞,∞] d lim k→0 V k dx (17
) = lim V→[-∞,∞] d lim k→0 Vk dx(18)
Therefore PDF corresponding to [0] can be written as:
PDF([0]) = lim V→[-∞,∞] d lim k→0 k Vk (19
) = lim V→[-∞,∞] d 1 V (20
)
Substituting Equation ( 18) into set (13) results in a set of constant values. From Equations ( 15) and ( 20), the PDF of the Metta-Class appears to follow a uniform distribution. This observation serves as an indication rather than a rigorous proof. If we consider the PDF of the Zero-Vector as a uniform distribution, and since it also satisfies the additive identity property, then in a peculiar sense, having data uniformly distributed across the entire space is equivalent to having no data at all. For a limited volume V, PDF corresponding to [0] is a uniform distribution with magnitude 1 V . Intuitively, if a threshold α f is used for the PDF f (x), then for f (x) + 1 V , using a threshold of α f + 1 V results in the exact same decision regions; therefore, the uniform distribution can be seen as the additive identity of the vector space. For the remainder of the report, the PDF of [0] is considered as uniform distribution. For an alternative explanation, see Section F in the Appendix. The differences in the learning outcomes of both networks are illustrated in the bottom plots of Figure 6. The bottomleft plot corresponds to the Zero-Exclusive-Network, while the bottom-right plot represents the Zero-Inclusive-Network. It can be observed that the boundaries learned by the Zero-Inclusive-Network are superior and clearer. By 'clear,' it is meant that decision regions of the Zero-Inclusive-Network overlaps exactly with the data. For instance, in the bottomleft plot of Figure 6, which illustrates the learning process of the Zero-Exclusive-Network, a significant portion of the empty space is misclassified as Class-0, Class-1, or Class-2.
this section cite: []

Section: UNARY/UNI-CLASS CLASSIFICATION
Imagine a scenario where all the collected data belongs to a single class. Is it still possible to train a neural network in such a case? Moreover, can the trained network determine whether a new data point belongs to this class or not? In neural networks, generally, the learning is done by comparing one class against another class(es). If the data belongs to only one class, say Class-0, the learning does not make much sense. This is where the Metta-Class is helpful. The data for the Metta-Class is a set of samples from uniform distribution. Now the original class-0 can be classified against the Metta-Class data with BCE loss using a neural network. The learning of the Neural Network trained on the combined data can be seen in the right subfigure of Figure 7. Therefore learning is possible even with only one class data, which is desirable since it is preferable to use thousands of smaller neural networks for training rather than training one giant neural network for a thousand classes.
Results for MNIST, CIFAR10 and ImageNet data are added in the Results Section E.1.2 of Appendix. 4.2.3. SET OPERATIONS ON THE CLASSES Upon closer examination, bottom-right plot of the Figure 6 exhibits resemblance to Venn Diagrams. This resemblance is no coincidence, as the addition 3.2.1 and scalar multiplication (with scalar value '-1') 3.2.2 defined on the Vector Space have the characteristics of set union and set complement respectively. As the set union (boolean addition) and set complement (boolean not) are the fundamental operations on sets (boolean logic), any composite set operation (composite boolean operation) can be written in terms of set union (boolean addition) and set complement (boolean not). The table 1 lists a few of the set operations on classes. Consider two classes, Class-1 and Class-2, trained on two different networks (uni-class classifiers) as described in Subsection 4.2.2. Let I c1 and I c2 denote the indicators (network outputs) for Class-1 and Class-2, respectively. The indicator values are 1 if the data belongs to the class they represent and 0 otherwise. Table 1 illustrates the logical operations on these classes. Furthermore, any boolean expression involving the classes can be evaluated using these logical operations. These logical operations can also be performed on multiple outputs of a single network. For instance, logical operations can be applied to the classes of the classifier corresponding to the bottom-right plot of Figure 6.
this section cite: []

Section: CONTINUAL LEARNING
In general, when the Metta-Class is not used to train neural network, every time a new class is added to the network, the network must be retrained or adopt some complicated techniques. For example say network-1 classifies all fe-
Table 1. Boolean operations between Class-c1 and Class-c2, where Ic 1 and Ic 2 are indicators derived from classifiers trained including the Metta-Class. Ic 1 = 1 implies the data point belongs to Class-c1, while Ic 1 = 0 implies it does not. For example, the fifth column represents the intersection operation, indicating whether a data point belongs to both Class-c1 and Class-c2, demonstrating that a separate classifier is not needed to identify data belonging to both classes.
I c1 I c2 I c2 I c1∪c2 I c1∩c2 I c1-c2 I c1⊕c2 I c1⊙c2 0 0 1 0 0 0 0 1 0 1 0 1 0 0 1 0 1 0 1 1 0 1 1 0 1 1 0 1 1 0 0 1
lines (cats, lions, tigers, cheetahs, jaguars etc) and if there is a need to classify pets (cats, dogs, parrots etc) a new neural network must be trained. The information about cats learned on the network-1 is not useful for pets classification. This kind of split brains among neural network does not integrate the learning. But if the Metta-Class is used in training, the knowledge can be integrated. Which means the information learned on old networks can be used for newer classifications. Over the time a set of networks will act as a repository, providing information for any kind of classification.
The reason is that the empty space (regions with no available data) is not classified as belonging to any class, as seen in the bottom-right plot of Figure 6. Consequently, the class boundaries learned do not create confusion with any new class, whose data may spread into the empty space of the old classifier's classes in the future. See Section B in the Appendix for an example. Neural network when trained properly with the Metta-Class, learns only the data manifold as shown in the Figure 7. It means that the logit values of the data from the class attain positive values only inside the manifold and negative value anywhere outside. If one hypothesizes that logit values attain peak inside the manifold, where the mode of the data lies, and gradually decreases when going away from the mode, then it is possible to reach the mode following the gradient of the logit hence generating a new synthetic datapoint in the process. Images in the Figure 9 are generated using simple gradient descent as shown in the equation 21, Class 0 Class 1 Class 2 Class 3 Class 4 Class 5 Class 6 Class 7 Class 8 Class 9 Figure 9. Each image is generated from a classifier trained on a single class (e.g., the bottom-far-left image is generated from a classifier trained solely on digit-5 MNIST data against the Metta-Class).
initialized with random noise image (x 0 ).
x k = x k-1 + α * ∂L ∂x x=x k-1 (21
)
where L is the logit of a particular class, as shown in the Figure 8 4.
2.6. EQUATION DISCOVERY Coefficients of Taylor series equation of a class can be discovered by treating Taylor series Equation (22) (23) as Logit equation of that class, with Cross Entropy loss on -10.0 -7.5 -5.0 -2.5 0.0 2.5 5.0 0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 Train Zero-Vector Class Class 1
Coefficients:
x = x -x 0 h(x) = m 0 + x T m 1 + x T m 2 x x 0 : [-2.
6516495 10.8534 ] m 0 : 3.091282367706299 m 1 : [ 0.19868742 -1.4092536 ] m 2 : [[-0.60970885 1.450288 ] [-1.3959112 -0.43427515]] whether the Logit value is positive or not.
h(x) = b + d j1=1 λ j1 x j1 + d j1=1 d j2=1 λ j1j2 x j1 x j2 + d j1=1 d j2=1 d j3=1 λ j1j2j3 x j1 x j2 x j3 + . . . (22
)
The equivalent Tensor equation for the Equation ( 22) can be written as below.
h( #» x ) = m 0 + m 1i x i + m 2ij x i x j + m 3ijk x i x j x k + ....(23)
where m 0 , m 1i , m 2ij , m 3ijk , ... are unknown scalars.
These unknown scalars are learned through backpropagation, as is done in neural networks. An empirical result illustrating this is shown in Figure 10. This can be used for discovering equations of lower dimensional physical phenomenons.
this section cite: []

Section: Metrics
To Identify which classifier is superior with respect to learning true boundary, we used the following two metrics:
Occupancy Factor = n [0]:C1 + n [0]:C2 + n [0]:C3 + ... N [0](24
) where N [0] is total data points of the Metta-Class used in training, and n [0]:C k is the data points of the Metta-Class recognized as class-C k .
this section cite: []

Section: Purity Factor Class
-C k = n (pred=C k ∧ label=C k ) n (pred=C k ) (25
)
where n (pred=C k ∧ label=C k ) is the total training examples that are predicted as class-C k and also labeled as class-C k , and n (pred=C k ) is the total training examples predicted as class-C k . Further details on these metrics are provided in Section D of the Appendix.
The applications discussed in this section are empirically evaluated in the Appendix. Detailed results and analysis can be found in Section E.
this section cite: []

Section: Time Complexity
To analyze the complexity clearly, we divide the computational requirements for any c-class classifier into two distinct parts: (1) Pre-softmax logit computation, and ( 2) Softmax computation. Let a Zero-Exclusive Network be a classifier that does not use the Metta-Class during training. Suppose there are n training points, and let the computation needed to compute pre-softmax logits be O(f (n))-for instance, f (n) might represent millions of computations. Computing the softmax function then takes O(n • softmax(c)) computations, where softmax(c) represents the complexity of softmax over c classes. Now consider modifying the Zero-Exclusive Network by adding an extra node at the output to classify the Metta-Class; we call this the Zero-Inclusive Network. In this scenario, we incorporate the Metta-Class in training. The addition of the new node introduces extra calculations dependent on the number of nodes in the preceding layer. Letting L denote the number of nodes in the layer preceding this new node, the additional computation required is O(nL 2 ).
The computational complexities during training and inference are summarized in the Table 2.
Typically, O(f (n)) dominates O(nL 2 ) since L corresponds only to the final internal layer. There is a trade-off between computational overhead and purity improvement. This tradeoff depends significantly on the intended applications leveraging the full potential of the Zero-Vector framework.
this section cite: []

Section: Related Work
Theorem 1 of Gutmann and Hyvärinen's pioneering work on Noise-Contrastive Estimation (2010) establishes that, under its conditions, the logit of a neural network trained to distinguish data sampled from a density p d (.) and a noise distribution p n (.) is given by equation 26
G(u; θ) = ln p d (u) -ln p n (u) (26
)
In our work, the noise distribution p n (.) is modeled as the Metta-Class, which follows a uniform distribution, i.e., p n (.) = k, where k is a known constant. Substituting this into equation 26, we obtain:
G(u; θ) = ln p d (u) -k(27)
If the expression ln p d (x)k is used as a representation in place of Valid Equivalence Sets (e.g., ln f (x)k for [f (x)], Table 2. Computational complexity comparison between a classifier that incorporates the Metta-Class data during training (referred to as a Zero-Inclusive Network) and one that does not (referred to as a Zero-Exclusive Network). Let the computation required to calculate pre-softmax logits during training for n data points be O(f (n)) in the Zero-Exclusive Network, where f (n) may represent millions of operations. During inference, the corresponding pre-softmax logit computation for m data points is denoted by O(g(m)).
this section cite: []

Section: Classifier Training Cost Inference Cost
Zero
-Exclusive Network O(f (n)) + O(n • softmax(c)) ≈ O(f (n)) O(g(m))+O(m•softmax(c)) ≈ O(g(m)) Zero-Inclusive Network O(f (n+k) + (n+k)L 2 ) + O((n+k) • softmax(c+1)) ≈ O(f (n+k)) O(g(m)+mL 2 )+O(m•softmax(c+1)) ≈ O(g(m)) ln g(x) -k for [g(x)
], ln u(x)k for [u(x)], etc.) of the Vector Space in section (3), the same conclusion can be reached-that is, the Metta-Class has a constant logit value. Consequently, the data in the Metta-Class can be interpreted as uniformly distributed.
Based on equation 27, the density can be expressed as:
p d (u) = e G(u;θ)+k(28)
and the score as:
∇ ln p d (u) = ∇G(u; θ)(29)
The score function derived from equation 29 has demonstrated its utility in generating new data using Langevin dynamics (Song & Ermon, 2019). This approach provides an explanation for the successful generation of MNIST data, as outlined in Subsection 4.2.5.
Furthermore, notable related works include advancements in Energy-Based Models, such as (LeCun et al., 2006) and (Grathwohl et al., 2019), which employ strategies to increase the energy of noise samples while decreasing the energy of true data samples.
Additional significant contributions in this domain include (Hinton, 2002), (van den Oord et al., 2018), and (Chen et al., 2020), which explore various methodologies for selfsupervised learning and contrastive representation learning.
this section cite: ['b16', 'b12', 'b6', 'b10', 'b18', 'b3']

Section: Conclusion
This work presents a mathematical framework to represent classes as vectors in a Vector Space, where Addition corresponds to the union of classes, and scalar multiplication closely resembles the set complement of classes. Notably, it was discovered that the Zero-Vector of the vector space is uniformly distributed in the feature space.
This framework opens up possibilities for developing several novel applications, many of which are not achievable with standard neural network training techniques. Additionally, this training methodology complements Noise Contrastive Estimation and Energy-Based Models, highlighting its utility in enriching existing paradigms.
However, this approach currently demonstrates limitations in scalability, as observed on CIFAR10 datasets. Addressing these challenges offers opportunities to extend this framework to handle higher-dimensional data effectively, paving the way for broader applicability and impact.
He deeply appreciates the support and collaboration of his lab-mates at VAL, IISc -Priyam Dey, Rishubh Parihar, Badrinath Singhal, Ankit Dhiman, and Abhipsa Basu -for their valuable technical discussions and suggestions.
He would also like to give special thanks to Mohd Shadab Ansari and Shatakshi Gupta for their consistent encouragement, helpful comments, corrections, and moral support throughout the project.
We would also like to thank the anonymous reviewers for their constructive comments and insightful suggestions that helped improve the quality of this work.
this section cite: []

Section: References
Ref_id:b0 Title: Linear algebra done right (eBook) Year: (2015)
Ref_id:b1 Title: Conditional noisecontrastive estimation of unnormalised models Year: (2018)
Ref_id:b2 Title: Anomaly detection: A survey Year: (2009-07)
Ref_id:b3 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b4 Title: Support-vector networks Year: (1995)
Ref_id:b5 Title: Implicit generation and modeling with energy based models Year: (2019)
Ref_id:b6 Title: Your classifier is secretly an energy based model and you should treat it like one Year: (2019)
Ref_id:b7 Title: Introduction to Tensor Analysis and the Calculus of Moving Surfaces Year: (2013)
Ref_id:b8 Title: Noise-contrastive estimation: A new estimation principle for unnormalized statistical models Year: (2010-05)
Ref_id:b9 Title: The forward-forward algorithm: Some preliminary investigations Year: (2022)
Ref_id:b10 Title: Training products of experts by minimizing contrastive divergence Year: (2002-08)
Ref_id:b11 Title: Survey on deep learning with class imbalance Year: (2019)
Ref_id:b12 Title: A tutorial on energybased learning Year: (2006)
Ref_id:b13 Title: Interpretable scientific discovery with symbolic regression: A review Year: (2024)
Ref_id:b14 Title: Efficient estimation of word representations in vector space Year: (2013)
Ref_id:b15 Title: A comprehensive survey of neural architecture search: Challenges and solutions Year: (2021-05)
Ref_id:b16 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
Ref_id:b17 Title: Ai feynman: a physicsinspired method for symbolic regression Year: (2020)
Ref_id:b18 Title: Representation learning with contrastive predictive coding Year: (2018)
Ref_id:b19 Title: When the PDF is constant Year: ()
Ref_id:b20 Title: When the PDF is strictly monotonically increasing. Case 1: PDF of the Zero-Vector is Constant A constant PDF corresponds to a uniform distribution. Hence, if the zero-vector has a constant PDF, the data associated with the zero-vector is uniformly distributed across its support. Case 2: PDF is Strictly Monotonically Increasing Let A be the volume spanned by a PDF that is strictly monotonically increasing and has zero probability outside the volume V . Since a PDF must integrate to one over its support, as A → ∞, the PDF appears approximately constant within any finite, localized region. Therefore, samples drawn from such a PDF will appear uniformly distributed in small local areas. From both Case 1 and Case 2, it follows that data sampled from the zero-vector appears uniform within any finite local region. Hence, the Metta-Class (the Zero-Vector Class) effectively behaves as a uniform distribution. NOTE: Some technical details Year: ()
