Title: Universal Causal Inference in a Topos
Abstract: In this paper, we explore the universal properties underlying causal inference by formulating it in terms of a topos. More concretely, we introduce topos causal models (TCMs), a strict generalization of the popular structural causal models (SCMs). A topos category has several properties that make it attractive: a general theory for how to combine local functions that define "independent causal mechanisms" into a consistent global function building on the theory of sheaves in a topos; a generic way to define causal interventions using a subobject classifier in a topos category; and finally, an internal logical language for causal and counterfactual reasoning that emerges from the topos itself. A striking characteristic of subobject classifiers is that they induce an intuitionistic logic, whose semantics is based on the partially ordered lattice of subobjects. We show that the underlying subobject classifier for causal inference is not Boolean in general, but forms a Heyting algebra. We define the internal Mitchell-Bénabou language, a typed local set theory, associated with causal models, and its associated Kripke-Joyal intuitionistic semantics. We prove a universal property of TCM, namely that any causal functor mapping decomposable structure to probabilistic semantics factors uniquely through a TCM representation.

Section: Introduction
In recent years, there has been significant interest in categorical models of causality, based on symmetric monoidal categories [Fong, 2012, Fritz and Klingler, 2023, Cho and Jacobs, 2019, Jacobs et al., 2018], as well as simplicial sets and higher-order categories [Mahadevan, 2023]. Markov categories [Fritz, 2020] define a broad unifying framework for probabilistic inference and statistics using symmetric monoidal categories, where each object is additionally equipped with a comonoidal "copy-delete" operation. It enables carrying out rigorous proofs using an elegant string diagrammatic language [Selinger, 2010]. Any causal model based on graphs [Pearl, 2009, Forré and Mooij, 2017, Spirtes et al., 2000] or other algebraic formalisms, such as integer-valued multisets [Studeny, 2010], can be translated into a string diagram over a symmetric monoidal category, or a simplicial set. Operations on causal models, such as interventions, can be modeled as functors on the objects of the associated symmetric monoidal category or simplicial set. Categorical approaches to causality also extend to the potential outcomes counterfactual framework [Imbens and Rubin, 2015].
Categorical approaches fundamentally differ from past work in causality in their focus on the elucidation of universal properties. In our previous work [Mahadevan, 2023[Mahadevan, , 2025c]], we introduced the framework of universal causality based on the notion of universal properties
Traffic Agricultural Fires Lung Infections Overpopulation Asthma Pollution Farming Practices Covid Lockdown
< l a t e x i t s h a 1 _ b a s e 6 4 = " T D Z 5 k X V c R i h T N / 4 4 I H 5 F t X D u l e A = " > A A A B 6 3 i c b V B N S 8 N A E J 3 U r 1 q / q h 6 9 L B b B U 0 l E W o 9 F L x 4 r 2 A 9 o Q 9 l s N 8 3 S 3 U 3 Y 3 Q g l 9 C 9 4 8 a C I V / + Q N / + N m z Y H b X 0 w 8 H h v h p l 5 Q c K Z N q 7 7 7 Z Q 2 N r e 2 d 8 q 7 l b 3 9 g 8 O j 6 v F J V 8 e p I r R D Y h 6 r f o A 1 5 U z S j m G G 0 3 6 i K B Y B p 7 1 g e p f 7 v S e q N I v l o 5 k l 1 B d 4 I l n I C D a 5 N E w i N q r W 3 L q 7 A F o n X k F q U K A 9 q n 4 N x z F J B Z W G c K z 1 w H M T 4 2 d Y G U Y 4 n V e G q a Y J J l M 8 o Q N L J R Z U + 9 n i 1 j m 6 s M o Y h b G y J Q 1 a q L 8 n M i y 0 n o n A d g p s I r 3 q 5 e J / 3 i A 1 4 Y 2 f M Z m k h k q y X B S m H J k Y 5 Y + j M V O U G D 6 z B B P F 7 K 2 I R F h h Y m w 8 F R u C t / r y O u l e 1 b 1 G v f F w X W v d F n G U 4 Q z O 4 R I 8 a E I L 7 q E N H S A Q w T O 8 w p s j n B f n 3 f l Y t p a c Y u Y U / s D 5 / A E X G o 5 K < / l a t e x i t > ω < l a t e x i t s h a 1 _ b a s e 6 4 = " H 4 Y V 6 L v Z L t P q r y G z l c F L 6 9 9 u 7 T s = " > A A A B 7 X i c b V B N S 8 N A E J 3 U r 1 q / q h 6 9 B I v g q S Q i 1 W P R i 8 c K 9 g P a U D a b T b t 2 s x t 2 J 0 I p / Q 9 e P C j i 1 f / j z X / j t s 1 B W x 8 M P N 6 b Y W Z e m A p u 0 P O + n c L a + s b m V n G 7 t L O 7 t 3 9 Q P j x q G Z V p y p p U C a U 7 I T F M c M m a y F G w T q o Z S U L B 2 u H o d u a 3 n 5 g 2 X M k H H K c s S M h A 8 p h T g l Z q 9 S I m k P T L F a / q z e G u E j 8 n F c j R 6 J e / e p G i W c I k U k G M 6 f p e i s G E a O R U s G m p l x m W E j o i A 9 a 1 V J K E m W A y v 3 b q n l k l c m O l b U l 0 5 + r v i Q l J j B k n o e 1 M C A 7 N s j c T / / O 6 G c b X w Y T L N E M m 6 W J R n A k X l T t 7 3 Y 2 4 Z h T F 2 B J C N b e 3 u n R I N K F o A y r Z E P z l l 1 d J 6 6 L q 1 6 q 1 + 8 t K / S a P o w g n c A r n 4 M M V 1 O E O G t A E C o / w D K / w 5 i j n x X l 3 P h a t B Exogenous variables < l a t e x i t s h a 1 _ b a s e 6 4 = " J z e G q J p o / n c 4 v O T s U G E f P 6 C m a y I = " > A A A B 6 3 i c b V B N S w M x E J 3 U r 1 q / q h 6 9 B I v g q e y K V I 9 F L x 4 r 2 A 9 o l 5 J N s 2 1 o k l 2 S r F C W / g U v H h T x 6 h / y 5 r 8 x 2 + 5 B q w 8 G H u / N M D M v T A Q 3 1 v O + U G l t f W N z q 7 x d 2 d n d 2 z + o H h 5 1 T J x q y t o 0 F r H u h c Q w w R V r W 2 4 F 6 y W a E R k K 1 g 2 n t 7 n f f W T a 8 F g 9 2 F n C A k n G i k e c E p t L g 8 T w Y b X m 1 b 0 F 8 F / i F 6 Q G B V r D 6 u d g F N N U M m W p I M b 0 f S + x Q U a 0 5 V S w e W W Q G p Y Q O i V j 1 n d U E c l M k C 1 u n e M z p 4 x w F G t X y u K F + n M i I 9 K Y m Q x d p y R 2 Y l a 9 X P z P 6 6 c 2 u g 4 y r p L U M k W X i 6 J U Y B v j / H E 8 4 p p R K 2 a O E K q 5 u x X T C d G E W h d P x Y X g r 7 7 8 l 3 Q u 6 n 6 j 3 r i / r D V v i j j K c A K n c A 4 + X E E T 7 q A F b a A w g S d 4 g V c k 0 T N 6 Q + / L 1 h I q Z o 7 h F 9 D H N y f R j l U = < / l a t e x i t > ω j Topos Causal Model Exogenous variables Endogenous variables Figure 1: Topos causal models (TCMs) are defined as a category C T CM whose objects c ∈ C T CM are causal models, and whose arrows C T CM (c, c ′ ) are commutative diagrams between models c and c ′ . A specific object c defining a model can be conceptualized as a DAG (left, where information flows from top to bottom), or a string diagram in a Markov category (middle, where information flows from bottom to top), or in terms of its induced unique "blackbox" function mapping exogenous variables to endogenous variables (right).
in category theory [Riehl, 2017]: a causal property is universal if it can be defined in terms of an initial or final object in a category of causal diagrams, or in terms of a causal representable functor using the Yoneda Lemma. For example, a structural causal model (SCM) [Pearl, 2009] is defined as a (deterministic) mapping from a collection of exogenous variables into a collection of endogenous variables, derived by "collating" local functions that serve as independent causal mechanisms [Galles andPearl, 1988, Parascandolo et al., 2017]. However, SCMs can be further analyzed in terms of their universal properties, such as categorical product, coproduct, limits and colimits, equalizers and coequalizers etc. These latter properties can be shown formally to be initial or final objects in a category of diagrams [Riehl, 2017], or as representable functors through the Yoneda Lemma [MacLane, 1971].
Our main contribution in this paper is to present a topos-theoretic view of causality, and in particular, introduce topos causal models (TCMs) that strictly generalize structural causal models (SCMs) [Pearl, 2009]. A topos is a type of category [MacLane, 1971], which is particularly well-suited to modeling operations that are "set-like" [MacLane and leke Moerdijk, 1994]. It also features an internal logical language [Goldblatt, 2006]. We claim that a topos provides three universal properties that make it natural as a category to do causal inference in: it provides a general theory for how to combine local functions, which can be viewed as "independent causal mechanisms" [Parascandolo et al., 2017], into a consistent global function building on the theory of sheaves in a topos [Mac Lane and Moerdijk, 1992]. It enables a generic way to define causal interventions using a subobject classifier in a topos category [Johnstone, 2014]. Finally, it gives an internal logical language for causal and counterfactual reasoning [Bell, 1988].
As Figure 1 illustrates, the objects in a TCM category can be conceptualized in multiple ways. First, each object can be a causal graphical model [Pearl, 1989, Spirtes et al., 2000]. Each object can also be a functor: for example, directed graphs form a topos functor category [Vigna, 2003]. TCMs can also be defined in terms of string diagrams in a symmetric monoidal Markov category [Fritz, 2020], where we restrict ourselves to the Markov subcategory defined through deterministic morphisms. For example, the arrow h : Traffic⊗Agricultural Fires → Pollution defines a deterministic mapping specifying the two potential causes of Pollution. For exogenous variables, the arrow ψ : I → Overpopulation defines the marginal distribution on Overpopulation, where I is the terminal object in the Markov category. Finally, we can view a TCM object as a "blackbox" function that maps some collection of exogenous variables (e.g., "Overpopulation", or "Farming Practices" into some set of endogenous variables, e.g., "Asthma" or "Pollution").
this section cite: ['b4', 'b8', 'b2', 'b13', 'b7', 'b30', 'b27', 'b6', 'b31', 'b32', 'b12', 'b29', 'b27', 'b9', 'b29', 'b17', 'b27', 'b17', 'b18', 'b10', 'b25', 'b16', 'b14', 'b0', 'b26', 'b31', 'b34', 'b7']

Section: Principles of Universal Causality
We give a brief overview of the fundamentals of universal causality (UC) [Mahadevan, 2023[Mahadevan, , 2025c] ] before delving into the specific details of the TCM framework. As with other work in categorical causality [Fong, 2012, Jacobs et al., 2018, Fritz and Klingler, 2023], UC uses category theory [MacLane, 1971] to define causality. A category C is a collection of abstract objects c ∈ C. Anything technically can count as an object, from a variable in a causal model to an entire model itself. Each category C is additionally specified by a set of arrows or morphisms C(c, d) between each pair of objects c and d. There is an identity arrow 1 c ∈ C(c, c). Arrows compose in the obvious way, inducing a function C(c, d) × C(d, e) → C(c, e). An initial object c in category C defined as one inducing a unique arrow from c to every object in category C. A terminal object, usually denoted by 1, is one that defines a unique arrow from every object c in category C into 1. An object c is isomorphic to another object d, denoted c ≃ d, if two arrows f : c → d and g : d → c exist, such that g • f = 1 c , and f • g = 1 d . A functor F : C → D between two categories C and D is specified by an object function mapping each c ∈ C to Fc ∈ D, and an arrows function mapping each arrow f ∈ C(c, d) to F f ∈ D(Fc, Fd). Functors come in two varieties -covariant and contravariant -the latter acts on the domain category by reversing the arrows. Given any two functors F : C → D and G : C → D between the same pair of categories, we can define a mapping between F and G that is referred to as a natural transformation. These are defined through a collection of mappings, one for each object c of C, thereby defining a morphism in D for each object in C.
this section cite: ['b4', 'b13', 'b8', 'b17']

Section: Fc
Gc
Fc ′ Gc ′ α c F f α c ′ G f
this section cite: []

Section: Yoneda Lemma and the Causal Reproducing Property
UC rests on the Yoneda Lemma -any object in a category can be defined by the interactions it makes with other objects (upto isomorphism). In the setting of causal inference, it means that objects in a TCM category can be ascribed "meaning" through studying the arrows of the category, without having to "look inside" the object. The Yoneda Lemma states that the set of all morphisms into an object d in a category C, sometimes denoted as Hom C (-, d), or as C(-, d), denoted as the presheaf, is sufficient to define d up to isomorphism. The category of all presheaves forms a category of functors, and is denoted Ĉ = Set C op . This category forms a topos, and will be fundamental to the TCM framework.
Lemma 1. [MacLane, 1971, Riehl, 2017] Yoneda lemma: For any functor F : C → Set, whose domain category C is "locally small" (meaning that the collection of morphisms between each pair of objects forms a set), and any object c in C, there is a bijection Hom(C(-, c), F) ≃ Fc that associates a natural transformation α : C(-, c) ⇒ F to the element α c (1 c ) ∈ Fc. This correspondence is natural in both c and F.
Definition 1. [Riehl, 2017] A universal property of an object c ∈ C in a category C is expressed by a representable functor F together with a universal element x ∈ Fc that defines a natural isomorphism C(-, c) ≃ F. The collection of morphisms C(-, c) into an object c is called the presheaf, and from the Yoneda Lemma, forms a universal representation of the object.
this section cite: ['b17', 'b29', 'b29']

Section: Diagrams and Universal Constructions
A key distinguishing feature of category theory is the use of diagrammatic reasoning. However, diagrams are also viewed more abstractly as functors mapping from some indexing category to the actual category. Diagrams are useful in understanding universal constructions, such as limits and colimits. Briefly, a diagram F : J → C is a functor F from some finite category J into a category of interest, C. For example, J = • → • ← • is an example of a "pullback" diagram. Here the • refer to abstract objects that are mapped into concrete objects in C by the functor F. What we want to know whether a particular diagram F or an entire class of diagrams is "solvable". What this means is whether its limit or colimit exists, that is, is the category complete or co-complete? For any object c ∈ C and any category J, the constant functor c : J → C maps every object j of J to c and every morphism f in J to the identity morphisms 1 c . We can define a constant functor embedding as the collection of constant functors ∆ : C → C J that send each object c in C to the constant functor at c and each morphism f : c → c ′ to the constant natural transformation, that is, the natural transformation whose every component is defined to be the morphism f . Definition 2. [Riehl, 2017] A cone over a diagram F : J → C with the summit or apex c ∈ C is a natural transformation λ : c ⇒ F whose domain is the constant functor at c. The components (λ j : c → F j) j∈J of the natural transformation can be viewed as its legs. Dually, a cone under F with nadir c is a natural transformation λ : F ⇒ c whose legs are the components (λ j : F j → c) j∈J . c F j Fk F j Fk c
λ j λ k F f F f λ j λ k
Cones under a diagram are referred to usually as cocones. Using the concept of cones and cocones, we can now formally define the concept of limits and colimits more precisely.
Definition 3. [Riehl, 2017] For any diagram F : J → C, there is a functor Cone(-, F) : C op → Set, which sends c ∈ C to the set of cones over F with apex c. Using the Yoneda Lemma, a limit of F is defined as an object lim F ∈ C together with a natural transformation λ : lim F → F, which can be called the universal cone defining the natural isomorphism C(-, lim F) ≃ Cone(-, F). Dually, for colimits, we can define a functor Cone(F, -) : C → Set that maps object c ∈ C to the set of cones under F with nadir c. A colimit of F is a representation for Cone(F, -). Once again, using the Yoneda Lemma, a colimit is defined by an object ColimF ∈ C together with a natural transformation λ : F → colimF, which defines the colimit cone as the natural isomorphism C(colimF, -) ≃ Cone(F, -).
Limit and colimits of diagrams over arbitrary categories can often be reduced to the case of their corresponding diagram properties over sets.
this section cite: ['b29', 'b29']

Section: The Universality of Diagrams and the Causal Reproducing Property
We state two key results that underly UC [Mahadevan, 2023]. While both these results follow directly from basic theorems in category theory, their significance for causal inference is what makes them particularly noteworthy. The first result pertains to the notion of diagrams as functors, and shows that for the functor category of presheaves, which is a universal representation of causal inference, every presheaf object can be represented as a colimit of representables through the Yoneda Lemma. This result can be seen as a generalization of the very simple result in set theory that each set is a union of one element sets. The second result is the causal reproducing property, which shows that the set of all causal effects between two objects is computable from the presheaf functor objects defined by them. Both these results are abstract, and apply to any category representation of a causal model. Theorem 1. [MacLane andleke Moerdijk, 1994, Mahadevan, 2023] Universality of Diagrams in UC: In the functor category of presheaves Set C op , every functor object F is the colimit of a diagram of representable objects, in a canonical way.
To explain the significance of this result for causal inference, note that UC represents causal diagrams as functors from an indexing category of diagrams to an actual causal model. The theorem above tells us that every presheaf object can be represented as a colimit of (simple) representable objects, namely functor objects of the form Hom C (-, c). Theorem 2. [MacLane, 1971, Mahadevan, 2023] Causal Reproducing Property: All causal influences between any two objects c and d can be derived from its presheaf functor objects, namely
Hom C (c, d) ≃ Nat(Hom C (-, c), Hom C (-, d))
Any causal influence of an object c upon any other object d can be represented as a natural transformation (a morphism) between two functor objects in the presheaf category Ĉ.
this section cite: ['b17', 'b17']

Section: Topos Causal Models
A topos [Johnstone, 2014] is a "set-like" category which generalizes all common operations on sets. Thus, the generalization of subset is a subobject classifier in a topos. To help build some intuition, consider how to define subsets without "looking inside" a set. Essentially, a subset S of some larger set T can be viewed as a "monic arrow" (an injective function).
this section cite: ['b14']

Section: Monic arrows generalize injective functions.
Definition 4. [Goldblatt, 2006] An arrow f : a → b in a category C is called monic if given any parallel arrows g, h : c → → a, the equality f • g = f • h implies that g = h, namely f is "left-cancellable".
Our approach builds on this abstraction to define a category C T CM whose objects are causal models, such as SCMs or Markov categories, and a submodel M x of an SCM M is simply a monic arrow f x : M x ↩→ M. Definition 5. A category C has binary products if for every pair of objects, c and d, there exists a third object, e ≃ c × d, along with two projection arrows, p 1 : e → c and p 2 : e → d, such that for any other object a and arrows f : a → c and g : a → d, there exists a unique morphism u : a → e satisfying p 1 • u = f and p 2 • u = g. Definition 6. [MacLane and leke Moerdijk, 1994] A category C with binary products has exponential objects if for each pair of objects c, d in C, there exists an object c d that defines the following bijection:
C(e × d, c) ≃ C(e, c d ) Definition 7. [MacLane and leke Moerdijk, 1994] A category C is Cartesian closed if it has binary products, a terminal object 1, and exponential objects.
Definition 8. [MacLane and leke Moerdijk, 1994] In a category C with finite limits, a subobject classifier is a C-object Ω, and a C-arrow true : 1 → Ω, such that to every monic arrow S ↩→ X in C, there is a unique arrow ϕ that forms the following pullback square:
S 1 X Ω m true ϕ
This commutative diagram enforces the constraint that every monic arrow m (i.e., every 1 -1 function) that maps a subobject S to an object X must be characterizable in terms of a "pullback", a particular type of universal property that is a special type of a limit. In the special case of the category of sets, subobject classifiers are defined through the characteristic (Boolean-valued) function ϕ that defines subsets. In general, as we show below, the subobject classifier Ω for causal models is not Boolean-valued, and requires using intuitionistic logic through a Heyting algebra. This definition can be rephrased as saying that the subobject functor is representable. In other words, a subobject of a causal model X in category C T CM is an equivalence class of monic arrows m : S ↩→ X. Definition 9. MacLane and leke Moerdijk [1994] An elementary topos is a category C that is Cartesian closed and has a subobject classifier.
For example, the category of sets forms a topos. Limits exist because one can define Cartesian products of sets, and colimits correspond to forming set unions. Exponential objects correspond to the set of all functions between two sets. Finally, the subobject classifier is simply the subset function, which induces a boolean-valued characteristic function.
Definition 10. The category C T CM of topos causal models is defined as a collection of objects c ∈ C T CM , each of which is a triple ⟨U, V, F⟩ where V = {V 1 , . . . , V n } is a set of endogenous variables, U is a set of exogenous variables, and F is a function from U to V. The arrows C T CM (c, d) are defined through commutative diagrams as illustrated below, where f and f ′ are the global functions induced by the TCM objects c and d, respectively.
U U ′ V V ′ h f f ′ g A submodel c ′ = ⟨U ′ , V ′ , F ′ ⟩ of c
is any subobject of c. The effect of an intervention on c is given by some submodel c ′ . Finally, let Y be a variable in V, and let X be a subset of V. The potential outcome in response to an intervention on X modeled by a submodel c ′ ↩→ c is the solution of Y in the submodel c ′ .
A commutative diagram, as the term suggests, is a structure showing the equivalence of two paths. Here, the diagram asserts that g • f = f ′ • h. In the context of our category C SCM , the arrow f : U → V is simply an SCM M, and f is its induced mapping from exogenous to endogenous variables. Similarly, f ′ is also the induced function mapping exogenous to endogenous variables for another SCM M ′ . The morphisms h and g are functions on SCMs, which transform one causal model into another. In the specific case we are interested in, these functions define causal interventions, but in general, they may be arbitrary functions.
For completeness, we define a category C SCM whose objects are indeed SCMs.
Definition 11. The category C SCM of structural causal models is defined as a collection of objects, each of which is a triple ⟨U, V, F⟩ where
V = {V 1 , . . . , V n } is a set of endogenous variables, U is a set of exogenous variables, F is a set { f 1 , . . . , f n } of "local functions" f i : U ∪ (V \ V i ) → V i whose composition induces a unique function F from U to V.
Let X be a subset of variables in V, and x be a particular realization of X. A submodel M x = ⟨U, V, F x ⟩ of M is the causal model M x = ⟨U, V, F x ⟩, where F x = { f i : V i X} ∪ {X = x}. The effect of an action do(X = x) on M is given by the submodel M x . Finally, let Y be a variable in V, and let X be a subset of V. The potential outcome of Y in response to an action do(X = x), denoted Y x (u), is the solution of Y for the set of equations F x .
The set of arrows or morphisms between two objects c and d in the category C SCM , denoted C SCM (c, d), represent ways of transitioning from SCM object c to d. For example, if d is a submodel of c, then the arrow defines a do calculus causal intervention.
this section cite: ['b10', 'b18', 'b18', 'b18']

Section: Causal Inference in a Topos Category
We show in this section that a TCM category whose objects are defined as SCMs, and whose arrows correspond to commutative diagrams defining operations on causal models does define a topos. In the next section, we generalize from SCMs to consider more complex causal models over functor categories. Now, we can state the first key result of this paper.
Theorem 3. The category C SCM forms a topos.
Proof: Since we have previously defined the objects and arrows of the C SCM category, to show it forms a topos, we need to construct its subobject classifier. First, we need to define what a "subobject" is in the category C SCM . Since SCMs can abstractly be defined as functions, let us assume that the SCM c that defines f is a submodel of the SCM c ′ that induces g. We can denote that by defining a commutative diagram as shown below. Let us stress the difference between the commutative diagram shown below Definition 10 for arbitrary functions g and h vs. the one below, where i and j are monic arrows.
U U ′ V V ′ i f g j P U" Q V" U V U' V' h m n f i j g q p u v U U' V V' {0} {0} {0, 1/2, 1}{0
9 i g u v M k = " > A A A B 7 X i c b V D L S g N B E O z 1 G e M r 6 t H L Y B A 8 h V 2 R 6 D H g
x W M E s w k k S 5 i d z C Z j 5 r H M z A p h y T 9 4 8 a C I V / / H m 3 / j J N m D J h Y 0 F F X d d H f F K W f G + v 6 3 t 7 a + s b m 1 X d o p 7 + 7 t H x x W j o 5 D o z J N a I s o r n Q n x o Z y J m n L M s t p J 9 U U i 5 j T d j y + n f n t J 6 o N U / L B T l I a C T y U L G E E W y e F P T An element x ∈ U ′ , which is a particular realization of the exogenous variables in U ′ , can be classified in three ways by defining a characteristic function ψ:
J i / b B f q f o 1 f w 6 0 S o K C V K F A s 1 / 5 6 g 0 U y Q S V l n B s T D f w U x v l W F t G O J 2 W e 5 m h K S Z j P K R d R y U W 1 E T 5 / N o p O n f K A C V K u 5 I W z d X f E z k W x k x E 7 D o F t i O z 7 M 3 E / 7 x u Z p O b K G c y z S y V Z L E o y T i y C s 1 e R w O m K b F 8 4 g g m m r l b E R l h j Y l 1 A Z V d C M H y y 6 s k v K w F 9 V r 9 / q r a a B R x l O A U z u A C A r i G B t x B E 1 p A 4 B G
1. x ∈ U -here we set ψ(x) = 1.
this section cite: []

Section: 2.
x U but g(x) ∈ V -here we set ψ(x) = 1 2 . 3. x U and g(x) V -we denote this by ψ(x) = 0.
The subobject classifier is illustrated as the bottom face of the cube shown on the right in Figure 2:
• true(0) = t ′ (0) = 1 • t : {0, 1 2 , 1} → {0, 1}, where t(0) = 0, t(1) = t( 1 2 ) = 1.
• χ V is the characteristic function of the exogenous variable set V.
• The base of the cube in Figure 2 displays the subobject classifier T : 1 → Ω, where T = ⟨t ′ , true⟩ that maps 1 = id {0} to Ω = t : {0, 1 2 , 1} → {0, 1}.
This proves that the subobject classifier for the category C SCM does not have Boolean semantics, but intuitionistic semantics as its subobject classifier Ω has multiple degrees of "truth", corresponding to the three types of classifications of monic arrows (in regular set theory, there are only two classifications). Moving on to show the other properties of a topos are satisfied, note that the terminal object is simply the identity function id 0 : {0} → {0}. Now, it remains to show that C SCM has pullbacks and exponential objects.
Pullbacks in C SCM : Consider the cube shown on the left in Figure 2. Here, f , g, and h can be interpreted as three SCMs, each mapping some exogenous variables to some endogenous variables. The arrows i, j ensure that the bottom face of the cube is a commutative diagram, and the arrows p, q ensures the right face of the cube is a commutative diagram. The arrow from P to Q exists because looking at the front face of the cube, Q is the pullback of i and q, which must exist because we are in the category of Sets, which has all pullbacks.
Similarly, the back face of the cube is a pullback of j and p, which is again a pullback in Sets. Summarizing, ⟨u, v⟩ and ⟨m, n⟩ are the pullbacks of ⟨i, j⟩ and ⟨p, q⟩.
Exponential objects in C SCM : Now it only remains to check that the category has exponential objects. Let f : U → V and g : U ′ → V ′ be two functions induced by SCM models M and N. Then, we need to define the meaning of g f in C SCM , which we can define as g f : X → Y, where Y = V ′V , which must exist since Sets is a Cartesian closed category that has exponential objects (i.e., Y is simply the set of all functions from V to V ′ ). Also, X is the set of all arrows in C SCM from SCM M to SCM N, which is the pair of functions ⟨h, k⟩ in the commutative diagram shown below. This finally proves that C SCM is a topos. □
this section cite: []

Section: Causal Models Over a Topos of Sheaves
We now describe a more general categorical framework for defining causal models as a topos by using the property that Yoneda embeddings of presheaves forms a topos [MacLane and leke Moerdijk, 1994]. To ensure consistent extension into a unique global function, we build on the theory of sheaves [Mac Lane and Moerdijk, 1992], which ensures local functions can be "collated" together to yield a unique global function. In our setting, we will construct sheaves from categories over causal models through the Yoneda embedding よ(x) : C → Sets C op and impose a Grothendieck topology.
this section cite: ['b18', 'b16']

Section: Grothendieck Topology on Sites
Definition 12. A sieve for any object x in any (small) category C is a subobject of its Yoneda embedding よ(x) = C(-, x). If S is a sieve on x, and h : y → x is any arrow in category C, then h * (S) = {g | cod(g) = D, hg ∈ S} Definition 13. [Mac Lane and Moerdijk, 1992] A Grothendieck topology on a category C is a function J which assigns to each object x of C a collection J(x) of sieves on x such that 1. the maximum sieve t
x = { f |cod( f ) = x} is in J(x).
2. If S ∈ J(x) then h * (S) ∈ J(y) for any arrow h : y → x.
3. If S ∈ J(x) and R is any sieve on x, such that h * (R) ∈ J(y) for all h : y → x, then R ∈ J(C).
We can now define categories with a given Grothendieck topology as sites. Definition 14. A site is defined as a pair (C, J) consisting of a small category C and a Grothendieck topology J on C.
Definition 15. The subobject classifier Ω is defined on any topos Sets C op as subobjects of the representable functors: Ω(x) = {S|S is a subobject of C(-, x)} and the morphism true : 1 → Ω is true(x) = x for any representable x.
this section cite: ['b16']

Section: Universal Property of TCM over Functor Categories
Causal models, like SCMs, must represent both decomposable structure and (probabilistic) semantics. To capture this richer structure, we define TCM over functor categories, where every object is a functor that maps structure to semantics. For example, the category of Bayesian networks can be modeled as a functor category [Jacobs et al., 2018, Fritz andKlingler, 2023] from a Markov category to the category FinStoch of finite stochastic processes. Theorem 4. Given a causal functor A : C → E, such as the Bayesian network functor F CDU , from a small category C (e.g., a symmetric monoidal category such as a Markov category) to a cocomplete category E (e.g., the category Prob of probability spaces (see Theorem 6)), the functor R from E to presheaves, given by (where c ∈ C and E ∈ E) R(E) : c → Hom E (A(c), E) has a left adjoint L : Sets C op → E defined for each presheaf P in C op as the colimit
L(P) = Colim C P π P --→ C A -→ E
where C P is the category of elements, whose objects are pairs (c, p), where c is an object of C and p is an element of P(C) (recall P is a presheaf, i.e., a set-valued functor that maps each element c into a set), and its arrows are (c ′ , p ′ ) → (c, p) for any morphism f : c ′ → c such that pc = p ′ .
Proof: Essentially, Theorem 4 is stating that there is a pair of adjoint functors L ⊢ R, defined as:
L : Sets C op → ← E : R
As defined earlier, a natural transformation between two functors τ : P → R(E) is a family {τ c } of maps indexed by the objects c ∈ C, where each map τ c is defined as the mapping: τ c : P(C) → Hom E (A(C), E) which is natural in c. τ can also be defined as a set of arrows of E as {τ c (p) : A(c) → E} (c,p) that is indexed by the objects (c, p) of the category C P of elements of P. This fact implies that there is a bijection Nat(P, R(E)) ≃ Hom E (LP, E) This bijection being natural in P and in E proves that L is a left adjoint functor to R. □ Now, let us define a general causal functor as mapping from a decomposable symmetric monoidal category (e.g., a Markov category) to the symmetric monoidal category of probability spaces. Definition 16. A causal functor F : C → Prob maps from a general symmetric monoidal category C with a comonoidal "copy-delete" structure (e.g., a CDU category [Jacobs et al., 2018] or a Markov category [Fritz, 2020]) to the category of probability spaces Prob, where each object (Ω, F , P) is a probability space, and the arrows are measure-preserving maps, namely Prob(c, d), where c = (Ω c , F c , P c ) and d = (Ω d , F d , P d ), where f ∈ Prob(c, d) is such that
P c ( f -1 (A)) = P d (A) for all A ∈ F d .
Theorem 5. For each causal functor A : C → E from a small category C defining the structure of a causal model to a cocomplete category E defining its (probabilistic) semantics, there exists a colimit preserving functor L : Sets C op → E such that A = L • よ, where よ is the Yoneda embedding.
Proof: The proof is just a special case of Corollary 4 on page 43 in [MacLane and leke Moerdijk, 1994]. To emphasize the importance of the co-completeness condition on E, we use the following result from [van Belle, 2024] that the category of probability spaces is co-complete. □ Theorem 6. [van Belle, 2024] The symmetric monoidal category Prob has all colimits of non-empty diagrams.
Proof: The proof, given in [van Belle, 2024], shows that Prob has coproducts and coequalizers. Thus, we can choose Prob as our cocomplete category E in Theorem 4. □
We can finally state the two central results of our paper, the first (Theorem 7) establishes the universal property underlying TCMs, and the second (Theorem 8) shows that causal interventions define a Heyting algebra whose logic is intuitionistic. Theorem 7. Any causal functor F : C → E from a structural causal category C (such as a Markov category) to a semantic cocomplete category E (such as Prob) factors uniquely through a TCM structure defined by the Yoneda embedding, as given in Theorem 5.
Proof: The proof follows directly from Theorem 4, Theorem 5, Theorem 6, and Definition 16. □ Definition 17. A Heyting algebra is a poset with all finite products and coproducts, which is Cartesian closed. That is, a Heyting algebra is a lattice, including bottom and top elements, denoted by 0 and 1, respectively, which associates to each pair of elements x and y an exponential y x . The exponential is written x ⇒ y, and defined as an adjoint functor: z ≤ (x ⇒ y) if and only if z ∧ x ≤ y
In other words, x ⇒ y is a least upper bound for all those elements z with z ∧ x ≤ y. As a concrete example, for a topological space X the set of open sets O(X) is a Heyting algebra.
The "law of the excluded middle", meaning ¬x ∨ x = true, does not always hold in a Heyting algebra.
Theorem 8. For any TCM category defined as Ĉ = Sets C op by the Yoneda embedding よ(c) of a small causal category C, the partially ordered set Sub Ĉ(P) of subobjects generated by causal interventions on any causal functor defined by the presheaf P is a Heyting algebra.
Proof: This result follows directly from the corresponding result for any category of presheaves (see [MacLane and leke Moerdijk, 1994]), and is based on constructing the complete lattice Sub(P) of all subfunctions of P using a pointwise operation for each object c ∈ C, which can be shown to satisfy an infinite distributive law. □
this section cite: ['b8', 'b13', 'b7', 'b18', 'b33', 'b33', 'b33', 'b18']

Section: Causal Mitchell-Bénabou Language and its Kripke-Joyal Semantics
The Causal Mitchell-Bénabou language (CMBL) is a typed local set theory [Bell, 1988] whose syntax and semantics is defined using the arrows of the C T CM topos. The types of CMBL as causal model objects M of C T CM . For each type M, we assume the existence of variables x M , y M , . . ., where each such variable has as its interpretation the identity arrow 1 : M → M. We can construct product objects, such as A × B × C, where terms like σ that define arrows are given the interpretation σ : A × B × C → D.
• Each variable x M of type M is a term of type M, and its interpretation is the identity x M = 1 : M → M, Here, M may represent an entire SCM, an individual variable such as Overpopulation in Figure 1, or a causal functor mapping a Markov category to the cocomplete Prob category.
• Terms σ and τ of types C and D that are interpreted as σ : A → C and τ : B → D can be combined to yield a term ⟨σ, τ⟩ of type C × D, whose joint interpretation is given as ⟨σp, τq⟩ : X → C × D, where X has the required projections p : X → A and q : X → B. A causal intervention modeled as an arrow f : X → Y in C TCM can be composed with a term σ : U → X to yield a term of type
Y as f • σ : U σ - → X f - → Y.
• Terms of type Ω are defined as formulae of CMBL and can be combined with the usual logical connectives ∧, ∨, ⇒, ¬ and quantifiers ∀, ∃ to obtain further terms again of type Ω. An expression such as ∀x ψ(x, y) is interpreted by an arrow Y → Ω (since x is not a free variable). A formula ψ(x, y) in the topos C TCM is defined to be universally valid in the topos if the corresponding arrow ψ(x, y) : X × Y → Ω factors through true : 1 → Ω. A formula ψ without free variables is interpreted as an arrow ψ : 1 → Ω and is valid if it coincides with the arrow true : 1 → Ω.
• Indirect proofs (i.e., reductio ad absurdum) cannot be used in CMBL because the rule of the excluded middle ψ ∨ ¬ψ is not in general valid, nor is the axiom of choice generally true. Instead, the rules of intuitionistic predicate calculus need to be used, as illustrated by de Araujo Fernandes and Haeusler [2009].
• The Kripke-Joyal semantics for CMBL is specified using generalized elements. We define an element of a causal model by the morphism x : 1 → M. Thus, a generalized element α : N → M represents the "stage of definition" of M by N. We specify the semantics of how an TCM model N supports any formula ϕ(α), denoted by N ⊩ ϕ(α) by N ⊩ ϕ(α) if and only if Im α ≤ {x|ϕ(x)}. Stated in the form of a commutative diagram, this "forcing" relationship holds if and only if α factors through {x|ϕ(x)}. See [MacLane andleke Moerdijk, 1994, Bell, 1988] for additional details.
{x|ϕ(x)} 1 N M Ω true α ϕ(x)
this section cite: ['b0', 'b17']

Section: Limitations and Future Work
There are significant limitations of our TCM framework, which we are currently investigating.
We have recently developed an intuitionistic generalization of Pearl's do-calculus termed j-stable causal inference, which uses the Lawvere-Tierney topology on a topos defined by a modal operator j on the subobject classifier Ω [Mahadevan, 2025a]. In this paper, we define an intuitionistic logic called j-do-calculus, where we replace global truth with local truth defined by Kripke-Joyal semantics. We are currently working on another paper [Mahadevan, 2025b] that implements j-do-calculus with well-known causal discovery procedures (e.g., score-based and constraint-based methods) [Zanga and Stella, 2023], and will include experimental results on how to (i) form data-driven j-covers (via regime/section constructions), (ii) compute chartwise conditional independences after graph surgeries, and (iii) glue them to certify the premises of the j-do rules in practice.
this section cite: ['b35']

Section: Acknowledgments
This research has been funded by Adobe Corporation.
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] Justification: The abstract and introduction clearly state the claims made, including the contributions made in the paper and important assumptions and limitations.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations.
A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: We include the limitation discussion in the paper in Section 7.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speechto-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: We give the assumptions and complete proofs in the main body of the paper.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [NA]
Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [NA]
Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.  cc/public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation,
including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [NA]
Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [NA]
Justification: The paper does not include experiments.
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
Answer: [NA]
Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: All authors have reviewed and confirmed that the research conducted in the paper conforms, in every respect, with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [Yes] Justification: We include the broader impacts discussion in the paper in Section F.
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
Justification: The paper does not include experiments and poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [NA]
Justification: The paper does not use existing assets.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/  datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16.
Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: Toposes and Local Set Theories Year: (1988)
Ref_id:b1 Title: Reinforcement Learning and Optimal Control Year: (2019)
Ref_id:b2 Title: Disintegration and bayesian inversion via string diagrams Year: (2019-03)
Ref_id:b3 Title: A topos-theoretic approach to counterfactual logic Year: (2009)
Ref_id:b4 Title: Causal theories: A categorical perspective on bayesian networks. Master's thesis Year: (2012)
Ref_id:b5 Title: Backprop as functor: A compositional perspective on supervised learning Year: (2019)
Ref_id:b6 Title: Markov properties for graphical models with cycles and latent variables Year: (2017)
Ref_id:b7 Title: A synthetic approach to markov kernels, conditional independence and theorems on sufficient statistics Year: (2020-08)
Ref_id:b8 Title: The d-separation criterion in categorical probability Year: (2023)
Ref_id:b9 Title: An axiomatic theory of counterfactuals Year: (1988)
Ref_id:b10 Title: The Categorial Analysis of Logic Year: (1974)
Ref_id:b11 Title: Categories for Quantum Theory: An Introduction Year: (2019-11)
Ref_id:b12 Title: Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction Year: (2015)
Ref_id:b13 Title: Causal inference by string diagram surgery Year: (2018)
Ref_id:b14 Title:  Year: (2014)
Ref_id:b15 Title: Counterfactuals and comparative possibility Year: (1973)
Ref_id:b16 Title: Sheaves in Geometry and Logic a First Introduction to Topos Theory Year: (1992)
Ref_id:b17 Title: Categories for the Working Mathematician Year: (1971)
Ref_id:b18 Title: Sheaves in Geometry and Logic: A First Introduction to Topos Theory Year: (1994)
Ref_id:b19 Title: Universal causality Year: ()
Ref_id:b20 Title: Intuitionistic j-do-calculus in topos causal models Year: (2025)
Ref_id:b21 Title: j-stable causal discovery in sites, 2025b Year: ()
Ref_id:b22 Title: Higher algebraic k-theory of causality Year: (2025)
Ref_id:b23 Title: Simplicial Objects in Algebraic Topology Year: (1992)
Ref_id:b24 Title: Umap: Uniform manifold approximation and projection for dimension reduction Year: (2018)
Ref_id:b25 Title: Learning independent causal mechanisms Year: (2017)
Ref_id:b26 Title: Probabilistic reasoning in intelligent systems -networks of plausible inference. Morgan Kaufmann series in representation and reasoning Year: (1989)
Ref_id:b27 Title: Causality: Models, Reasoning and Inference Year: (2009)
Ref_id:b28 Title: Cambridge Studies in Advanced Mathematics Year: (2020)
Ref_id:b29 Title: Category Theory in Context. Aurora: Dover Modern Math Originals Year: (2017)
Ref_id:b30 Title: A survey of graphical languages for monoidal categories Year: (2010)
Ref_id:b31 Title: Adaptive computation and machine learning Year: (2000)
Ref_id:b32 Title: Probabilistic Conditional Independence Structures. Information Science and Statistics Year: (2010)
Ref_id:b33 Title: Kan Extensions in Probability Theory Year: (2024)
Ref_id:b34 Title: A guided tour in the topos of graphs Year: (2003)
Ref_id:b35 Title: A survey on causal discovery: Theory and practice Year: (2023)
