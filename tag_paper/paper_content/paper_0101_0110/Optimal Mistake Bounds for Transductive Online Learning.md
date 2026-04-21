Title: Optimal Mistake Bounds for Transductive Online Learning
Abstract: We resolve a 30-year-old open problem concerning the power of unlabeled data in online learning by tightly quantifying the gap between transductive and standard online learning. In the standard setting, the optimal mistake bound is characterized by the Littlestone dimension d of the concept class H (Littlestone, 1987). We prove that in the transductive setting, the mistake bound is at least Ω √ d . This constitutes an exponential improvement over previous lower bounds of Ω(log log(d)), Ω log(d) , and Ω(log(d)), due respectively to Ben-David, Kushilevitz, and   Mansour (1995, 1997), and Hanneke, Moran, and Shafer (2023). We also show that this lower bound is tight: for every d, there exists a class of Littlestone dimension d with transductive mistake bound O √ d . Our upper bound also improves upon the best known upper bound of (2/3) • d from Ben-David et al. (1997). These results establish a quadratic gap between transductive and standard online learning, thereby highlighting the benefit of advance access to the unlabeled instance sequence. This contrasts with the PAC setting, where transductive and standard learning exhibit similar sample complexities.

Section: Introduction
The transductive model is a basic and well-studied framework in learning theory, dating back to the early works of Vapnik. It has been investigated both in statistical and online settings, and is motivated by the principle that to make good predictions on a specific set of test instances, one need not construct a fully general classifier that performs well on the entire domain -including points that may never actually appear. Rather, it may be sufficient to tailor predictions for a fixed, known set of instances.
This perspective naturally connects to a broader question in learning theory: what is the value of unlabeled data? In the transductive setting, the learner is given the sequence of unlabeled test instances in advance and is then required to predict their labels one by one. Thus, the transductive model can be viewed as a natural formalization of learning with unlabeled data: the test instances are known in advance, but their labels are not. The central question is whether such prior access to the unlabeled sequence can help reduce the number of prediction mistakes -compared to the standard online model, where the instances arrive and are labeled one at a time.
Recall for instance that in the standard PAC 1 model of supervised learning, there are cases where access to unlabeled data is not helpful. Indeed, the "hard population distributions" used to prove the standard VC 2 lower bound are constructed by taking a fixed and known marginal distribution over a VC-shattered set. Namely, the cases that are hardest to learn in the PAC setting include ones where the learner knows the marginal distribution over the domain, and can therefore generate as much unlabeled data as it wishes. And yet, in those cases, access to unlabeled data provides no acceleration compared to an algorithm (like ERM 3 ) that does not use unlabeled data.
Seeing as unlabeled data is often a lot easier to obtain than labeled data, there have been considerable efforts to understand when and to what extent can access to unlabeled data accelerate learning. 4   In particular, it is natural to ask, for which plausible models of learning is access to unlabeled data beneficial? Online learning (Littlestone, 1987) is perhaps the model of learning that is mostextensively studied in learning theory after the PAC model and its variants. Therefore, the general question considered in this paper is: Question 1. Quantitatively, how much (if at all) is access to unlabeled data beneficial for learning in the online learning setting?
This question is naturally instantiated by comparing transductive online learning -where the learner has advance access to the full sequence x 1 , x 2 , . . . , x n of unlabeled instances -with standard online learning, where no such access is given. This perspective has also been adopted in prior work: for example, Kakade and Kalai (2005), Cesa-Bianchi and Shamir (2013), and Hoi, Sahoo, Lu, and Zhao (2021) (Section 7.3) all describe transductive online learning as a setting in which the learner has access to "unlabeled data". We thus refine the question above as follows: Question 2. Quantitatively, how much (if at all) is learning in the transductive online learning setting easier than learning in the standard online learning setting? Specifically, how much is the optimal number of mistakes in the transductive setting smaller than in the standard setting?
Addressing this question, our main result (Theorem 1.1) states that the optimal number of mistakes in the transductive setting (with access to unlabeled data) is at most quadratically smaller than in the standard setting (without unlabeled data). Furthermore, there are hypothesis classes for which a quadratic gap is achieved.
this section cite: ['b18', 'b17', 'b7']

Section: Setting: Standard vs. Transductive Online Learning
Standard online learning (Littlestone, 1987) is a zero-sum, perfect-and complete-information game played over n rounds between two players, a learner and an adversary. The game is played with respect to a domain set X and a hypothesis class H ⊆ {0, 1} X (consisting of functions X → {0, 1}), where n, X and H are fixed and known to both players. The game proceeds as in Game 1. The number of mistakes for a learner L and an adversary A is M std (H, n, L, A) = |{t ∈ [n] : ŷt ̸ = y t }|. We are interested in understanding the optimal number of mistakes, which is
M std (H) = sup n∈N inf L∈L sup A∈A M std (H, n, L, A),
where A and L are the set of all deterministic adversaries and learners, respectively. 51 Probably Approximately Correct. For an exposition of the standard terminology and results mentioned in this paragraph see, e.g., Shalev-Shwartz and Ben-David (2014). 2 Vapnik-Chervonenkis. 3 Empirical Risk Minimization. 4 The literature on semi-supervised learning is surveyed in Joachims (1999); Zhu (2005); Zhu and Goldberg (2009); Zhu (2010); Chapelle, Schölkopf, and Zien (2006). Theoretical works on the topic include Benedek and Itai (1991); Blum and Mitchell (1998); Ben-David, Lu, Pál, and Sotáková (2008); Balcan and Blum (2010); Darnstädt, Simon, and Szörényi (2013); Göpfert, Ben-David, Bousquet, Gelly, Tolstikhin, and Urner (2019). 5 Because the adversary selects yt after seeing ŷt, randomness is not beneficial for either party, and we assume without loss of generality that both the learner and the adversary are deterministic. As is common in learning theory, we avoid questions of computability and allow the learner and adversary to be any function. See Section A for formal definitions of A and L.
For each round t = 1, 2, . . . , n: a. The adversary selects an instance x t ∈ X and sends it to the learner. b. The learner selects a prediction ŷt ∈ {0, 1} and sends it to the adversary. c. The adversary selects a label y t ∈ {0, 1} and sends it to the learner. The selected label must be realizable, meaning that ∃h ∈ H ∀i ∈ [t] : h(x i ) = y i .
Game 1: The standard online learning setting.
The adversary selects a sequence x 1 , x 2 , . . . , x n ∈ X and sends it to the learner.
For each round t = 1, 2, . . . , n:
a. The learner selects a prediction ŷt ∈ {0, 1} and sends it to the adversary.
b. The adversary selects a label y t ∈ {0, 1} and sends it to the learner. The selected label must be realizable, meaning that ∃h ∈ H ∀i ∈ [t] : h(x i ) = y i .
Game 2: The transductive online learning setting.
It is well known that M std (H) is characterized by the the Littlestone dimension, namely, M std (H) = LD(H) (see Theorem A.7 and Definition A.6).
The transductive online learning setting (Ben-David et al., 1995, 1997) is similar, except that the learner has access to the full sequence of unlabeled instances in advance. Namely, as in Game 2. The optimal number of mistakes for the transductive setting is defined exactly as before, M tr (H, n, L, A) = |{t ∈ [n] : ŷt ̸ = y t }|, and M tr (H) = sup n∈N inf L∈L sup A∈A M tr (H, n, L, A), with the only difference between the standard quantity M std (H) and the transductive quantity M tr (H) being in how the game is defined.
this section cite: ['b18', 'b19', 'b16', 'b22', 'b4', 'b5', 'b3', 'b0', 'b10', 'b12', 'b1', 'b2']

Section: Main Result
Notice that for every hypothesis class H, M tr (H) ≤ M std (H). Indeed, in the transductive setting the adversary declares the sequence x at the start of the game. This reduces the number of mistakes because the transductive adversary is less powerful (it cannot adaptively alter the sequence mid-game), and also because the transductive learner is more powerful (it has more information). 6While for some classes M tr (H) = M std (H), we study the largest possible separation. The best previous lower bound on M tr , due to Hanneke, Moran, and Shafer (2023), states that for every class H, M tr (H) ≥ Ω(log(d)), where d = M std (H). In the other direction, Ben-David et al. (1997) constructed 7 a class H such that M std (H) = d and M tr (H) ≤ 2 3 d. This left an exponential gap between the best known lower and upper bounds on M tr , namely Ω(log d) versus 2 3 d. Our main result closes this gap: Theorem 1.1 (Main result).
• For every hypothesis class H ⊆ {0, 1} X , M tr (H) = Ω √ d , where d = M std (H).
• On the other hand, for every d there exists a hypothesis class H with M std (H) = d and
M tr (H) = O √ d .
This result is stated in considerably greater detail in Theorems B.1 and D.1.
this section cite: ['b13', 'b2']

Section: Related Works
The notion of transductive inference as a more efficient alternative to inductive inference in statistical learning theory was introduced by Vapnik (1979Vapnik ( , 2006)); Gammerman, Vovk, and Vapnik (1998); Chapelle, Vapnik, and Weston (1999). The online learning setting is due to Littlestone (1987), who also proved that the optimal number of mistakes is characterized by the Littlestone dimension (see Theorem A.7).
The transductive online learning setting studied in the current paper, was first defined by Ben-David, Kushilevitz, and Mansour (1995), who used the name worst sequence off-line model. Among other results, they showed a lower bound of Ω(log log(d)) on the number of mistakes required to learn a class with Littlestone dimension d. The authors subsequently presented an exponentially stronger lower bound of Ω log(d) in Ben-David, Kushilevitz, and Mansour (1997). However, understanding
where the optimal number of mistakes is situated within the range Ω log(d) , 2d/3 remained an open question. Kakade and Kalai (2005) presented an oracle-efficient algorithm for the transductive online learning setting, and may have been the first to use that name. Their result was subsequently improved upon by Cesa-Bianchi and Shamir (2013).
The present work is most similar to that of Hanneke, Moran, and Shafer (2023) which, among other results, gave a quadratically-stronger mistake lower bound of Ω(log(d)) for classes with Littlestone dimension d in the transductive online setting. The proof of our lower bound utilizes some of their ideas, but yields a quantitative improvement by combining it with some new ideas.
Hanneke, Raman, Shaeiri, and Subedi (2024) studied a setting of multi-class transductive online learning where the number of possible labels is unbounded.
this section cite: ['b20', 'b21', 'b11', 'b8', 'b18', 'b1', 'b2', 'b17', 'b7', 'b13']

Section: Technical Overview
In this section we explain some of the main ideas in our proofs. Formal definition appear in Section A. Full formal statements of the results, as well as detailed rigorous proofs, appear in Sections B to D.
this section cite: []

Section: Paths in Trees
We make extensive use of the following notion. Given a perfect binary tree T d of depth d, every function f : T d → {0, 1} defines a unique path in the tree. The path is a sequence of nodes path(f ) = (x i0 , x i1 , . . . , x i d ), as explained in Figure 1c. See Section A for formal definitions.
this section cite: []

Section: Proof Ideas for the Lower Bound
We start with an elementary observation about the adversary's dilemma in the transductive online learning setting. Before round t of the game, the adversary selected a full sequence of instances x 1 , x 2 , . . . , x n ∈ X , and assigned some initial labels y 1 , y 2 , . . . , y t-1 ∈ {0, 1}. At the start of round t, the adversary must consider the version space,
H t = h ∈ H : (∀i ∈ [t -1] : h(x i ) = y i ) .
If all h ∈ H t assign h(x t ) = b for some b ∈ {0, 1}, then the adversary has no choice but to assign the label y t = b. Otherwise, the adversary can force a mistake at time t. Namely, after seeing the learner's prediction ŷt , the adversary can assign y t = 1ŷt , incrementing the number of learner mistakes by 1.
But "just because you can, doesn't mean you should". If the adversary is greedy and forces a mistake at time t, they may pay dearly for that later. As an extreme example, consider the case where there
x 0 x 1 x 3 x 4 x 2 x 5 x 6
(a) A perfect binary tree of depth 2. Each node is labeled by an element of the domain X . These labels need not be distinct (e.g., it is possible that x1 = x6). x0 is the root of the tree, x0, x1 and x2 are internal nodes, and x3, . . . , x6 are leaves.
x 0
x 1 x 3 • • x 4 • • x 2 x 5 • • x 6 • • 0 0 0 1 1 0 1 1 0 0 1 1 0 1 (b) A function f : X → {0, 1}
assigns a binary label to each node in the tree, represented here by edges with arrowhead tips. This figure depicts the function f (xi) = 1(i / ∈ {2, 3}). (Note that the gray dots (•) in the figure are purely a pictorial detail. In this paper they are not considered nodes or leaves of the tree.)
x 0 x 1 x 3 • • x 4 • • x 2 x 5 • • x 6 • • 0 0 0 1 1 0 1 1 0 0 1 1 0 1 (c) Every function f : X → {0, 1} defines a path in the tree, which is a sequence u0, u1, u2, . . . , u d-1 ,
where u0 is the root, d is the depth of the tree, and for each i ∈ [d -1], ui is the b-child of ui-1 with b = f (ui-1) ∈ {0, 1}. This figure shows that the function f from Figure 1b has path(f ) = (x0, x2, x5), depicted in red. In particular, x2 is 'on-path' for f , but x6 is 'off-path' for f . λ 0 00
• • 01 • • 1 10 • • 11 • • 0 0 0 1 1 0 1 1 0 0 1 1 0 1 (d)
In this paper we use a naming convention where, without loss of generality, we identify the domain elements xi that are assigned to nodes with bit strings. The root is identified with the empty string λ, and for each pair of nodes u, v such that u is the b-child of v (for b ∈ {0, 1}), we have u = v • b, where '•' denotes string concatenation. (Because the xi's may not be distinct, a domain element may be identified with more than one bit string.) is a single h 1 ∈ H t that assigns h 1 (x t ) = 1, and all other functions h ∈ H t assign h(x t ) = 0.
If the learner selects ŷt = 1 and the adversary forces a mistake at time t, the version space at all subsequent times s > t will be H s = {h 1 }, and the adversary will be prevented from forcing any further mistakes.
A natural strategy for the adversary is therefore to be greedy up to a certain limit. Namely, at each time t the adversary computes the ratiofoot_2
r t = |{h ∈ H t : h(x t ) = 1}| |H t | .
If r t ∈ [ε, 1ε] for some parameter ε > 0 ("the version space is not too unbalanced"), then the adversary forces a mistake. Otherwise, the adversary assigns the majority label, i.e., y t = 1(r t ≥ 1/2). This ensures that the version space does not shrink too fast:
• If no mistake is forced, then
|H t+1 | ≥ (1 -ε) • |H t |, and • If a mistake is forced, |H t+1 | ≥ ε • |H t |.
In particular, at the end of the game, the version space H n+1 is of size
|H n+1 | ≥ ε M • (1 -ε) n-M • |H| ≥ ε M • (1 -ε) n • 2 d , (1
)
where M is the number of mistakes that the adversary forces and n is the length of the sequence. The class has size |H| ≥ 2 d because LD(H) = d, and by removing functions from the class if necessary (which can only make learning easier), we may assume without loss of generality that |H| = 2 d . Namely, the class precisely shatters a Littlestone tree of depth d -1 such that for every assignment of labels to a root-to-leaf path in the tree, the class contains exactly one function that agrees with that assignment (see Definition A.6 for detail).
Notice that we have not yet specified how the adversary selects the sequence x. While the adversary's labeling strategy is extremely simple (determined by the ratio r t and the prediction ŷt ), constructing of the sequence x requires some care, to ensure that it has the following two properties:
• Property I: The length n of the sequence satisfies n = 2 Θ( d) , and
√
• Property II: For every sequence of predictions ŷ1 , . . . , ŷn selected by the learner, the resulting sequence of labels y 1 , . . . , y n selected by the adversary are consistent with some function h ∈ H such that x contains all the nodes in path(h).foot_3
These properties can be achieved by carefully simulating all possible execution paths of the adversary.
Observe that if path(h) = (u 1 , . . . , u d ) then the sequence of labels h(u 1 ), h(u 2 ), . . . , h(u d ) uniquely identifies the function h within the class H. Hence, Property II and the assumption |H| = 2 d imply that at the end of the game, the version space H n+1 has cardinality
|H n+1 | = 1.(2)
Combining Property I (n = 2 Θ( √ d) ), Eqs. (1) and ( 2), and choosing ε = 2 -Θ(
√ d) gives 1 ≥ ε M • (1 -ε) n • 2 d ≥ 2 -Θ(M • √ d) • 2 d ,
which implies M = Ω √ d , as desired.
this section cite: []

Section: Proof Ideas for the Upper Bound
In this section we explain the main ideas in the proof of Theorem D.1, which states that for every d ∈ N, there exists a class of Littlestone dimension d that is learnable in the transductive online setting with a mistake bound of O √ d .
Of course, not every Littlestone class satisfies this property. For instance, the set of all functions [d] → {0, 1} has Littlestone dimension d, but the adversary can force the learner to make d mistakes when learning this class in the transductive setting. 10 So our task in this proof is to construct a class that is especially easy to learn in the transductive setting (i.e., learnable with O √ d mistakes), while still being hard (requiring d mistakes) in the standard setting.
this section cite: []

Section: Sparse Encodings are Easy to Guess
We start with an elementary observation. Consider the following two bit strings: Binary: 110101 One-hot: 0000000000000000000000000000000000000000000000000000100000000000 Both of these strings encode the number 53. However, one of the encodings is much easier to guess than the other: suppose we are tasked with guessing the bits in an encoding of an integer between 0 and 2 6 -1. We guess the bits one at a time, and after each guess, an adaptive adversary tells us whether our guess was correct. Now, if the bit string is a binary encoding, the task is hard. Each bit can either be 0 or 1, regardless of the values of the previous bits, and so the adversary can force a mistake on every bit. On the other hand, if we know that the string is a one-hot encoding, there exists an attractive strategy -always guess 0. This ensures that we will make at most 1 mistake.
Note that at the end of the guessing game we have learned the same amount of information (for a number between 0 and 2 n -1, we learned n bits of information), but the number of mistakes is very different (n mistakes vs. 1 mistake).
this section cite: []

Section: Construction of the Hypothesis Class
We now describe a construction of a hypothesis class that is easy to learn in the transductive setting, using the idea of a sparse encoding. Recall that a class H has Littlestone dimension at least d (Definition A.6 in Section A) if there exists a Littlestone tree of depth d -1 such that for every b ∈ {0, 1} d there exists h = h b ∈ H such that the values on the path of h agree with b. More formally, ∀i ∈ [d] : h(b <i ) = b i , and in particular path(h) = (λ, b ≤1 , b ≤2 , b ≤3 , . . . , b ≤d-1 ). Thus, when constructing a class that shatters a specific Littlestone tree of depth d -1, we need to define 2 d functions h b : b ∈ {0, 1} d . For each function h b , the on-path values of the function are fixed (fully determined by b), while for the remaining values there is complete freedom (for the nodes u that are off-path we may assign any values h b (u) ∈ {0, 1}).
Perhaps the simplest way to construct a class of Littlestone dimension d is simply to assign all on-path values as required, and assign 0 to all other values. Namely, if u is a prefix of b then h b (u) = b |u|+1 , and otherwise h b (u) = 0. In a sense, this is the 'minimal' class of Littlestone dimension d for a specific Littlestone tree. 11   Observe that the 'minimal' class does not have the desired property of being easy to learn in the transductive setting. 12 However, a certain variation of the 'minimal' class that embeds a sparse encoding does satisfy the requirement. In this variation, on-path value of the function h b are assigned as they must (as determined by b), while the off-path values are sampled independently using a biased coin, such that each of them is 0 with high probability, but has a small probability of being 1. The probability is chosen carefully so that the class satisfies some simple combinatorial properties, as described further in Section 2.3.6 and Lemma D.2.
this section cite: []

Section: Naïve Learning Strategy
We now explain in broad strokes how the probabilistic construction of the hypothesis class in Section 2.3.2 is useful for learning with few mistakes in the transductive setting.
Notice that when predicting labels for the 'minimal' class with nodes in breadth-first order, the learner knows at each step whether they are labeling an on-path or off-path node, because the learner has already seen the correct labels for all ancestors of the current node. For off-path nodes, the learner knows that the true label is 0, so it never makes mistakes on off-path nodes, but it also gains no new information when the true labels for off-path nodes are revealed. No risk, but no reward either. Instead, all the information about the true labeling function is revealed only at on-path nodes, where the adversary has complete freedom to assign labels and force mistakes. That's why the adversary can force d mistakes.
For the randomly-chosen class, when predicting labels for off-path nodes, the learner may still safely predict a label of 0. But the reasoning for this is quite different. Conceptually, every off-path label is part of a sparse codeword that identifies the correct labeling function. 13 Because the coin is biased, each bit of the codeword is easy to guess (it is likely to be 0), but every time that the adversary reveals that the true label for an off-path node is indeed 0, the learner gains a small (nonzero) amount of 11 More formally, this is a class with a minimal number of nodes labeled 1. 12 The adversary can declare a sequence x consisting of all the nodes in the tree in breadth-first order, and then force d mistakes -one mistake in each layer (depth) of the tree. Specifically, regardless of how the adversary selects the labels, for each i ∈ [d] there exists a node ui at depth i that is on-path. When it is time for the learner to predict a label for this ui, the learner knows that ui is on-path because it has seen the correct labels for all the ancestors of ui. However, the adversary has the freedom to extend the path arbitrarily to the left or to the right, and can therefore force a mistake on ui.
13 The coin-flips for off-path labels are all independent. For example, if X is a set of nodes all of which are off-path for a subset H of the hypothesis class, then the random variables {h(x) : h ∈ H, x ∈ X} are i.i.d. information about the true labeling function. Additionally, when the adversary selects an off-path label of 1, that reveals a lot of information about the true labeling function (such labels are rare in the hypothesis class), and therefore the adversary cannot force many off-path mistakes. Overall, the information about the true labeling function is 'smeared' throughout all labels of the tree (0s and 1s, on-path and off-path). 14   Thus, the naïve general strategy for the learner when using the probabilistically-constructed class is to learn most of the information about the true labeling function by observing off-path labels. By the time the learner reaches an on-path node, it hopefully has already learned enough about the true labeling function in order to make a good prediction on that node. However, making this general strategy work requires overcoming some very substantial obstacles:
1. Recall that in the transductive setting, the adversary can present the nodes of the tree in any order of its choosing -it does not have to present the tree in breadth-first order. The naïve strategy works only if the learner sees many off-path nodes before it sees most on-path nodes. But what happens if the adversary decides to present many on-path nodes near the beginning of the sequence? To handle this, the learner incorporates a strategy we call 'danger zone minimization', as described in Section 2.3.4.
2. Another, equally problematic, issue also arises from the fact that the sequence presented by the adversary might not be in breadth-first order. Recall that breadth-first order 15 has the property that for every node u in the sequence, all the ancestors of u appear before u in the sequence. This means that by the time the learner needs to predict a label for u, the learner knows whether u is on-path or off-path for the true labeling function. But what happens if the adversary presents u before some of u's ancestors? Or omits some of u's ancestors from the sequence altogether? In this case the learner doesn't know if u is on-path or off-path, and this presents a double hazard. One hazard is that the leaner doesn't know what label to predict for u -if u is off-path, the learner can simply predict 0, but if it is on-path it must do something more elaborate. The second hazard is that, after seeing the correct label for u, it is not clear what the learner can infer from it. If u is off-path, its label should be interpreted as part of a sparse encoding of the labeling function. But if u is on-path, the interpretation must be entirely different. To overcome this challenge, the learner incorporates a strategy we call 'splitting experts', described in Section 2.3.5.
this section cite: []

Section: Limiting off-path mistakes.
Thanks to the coin's bias, most off-path nodes have a true label of 0. Nonetheless, each function in the hypothesis class still has an expected number of 2 Ω(d) off-path nodes labeled 1, so the learner can afford to misclassify only a vanishing fraction of them! To limit the number of mistakes, the learner extracts information from the sparse encoding and executes a 'transition to Halving' strategy, as described in Section 2.3.6.
this section cite: []

Section: Danger Zone Minimization
Utilizing information from the 'sparse encoding' of the off-path nodes to make good predictions for on-path nodes requires that the learner first see the true labels for many off-path nodes. Until that happens, the learner expects to make many mistakes on on-path nodes. However, whether a node is on-path or off-path is not fixed in advanced -the adversary may decide this adaptively, in response to the learners predictions.
Danger zone minimization is a strategy used by the learner, to force the adversary to assign few nodes in the beginning of the sequence as on-path (otherwise, if initial nodes are assigned to be on-path by the adversary, then the learner will make few mistakes on those nodes). This is analogous to the standard Halving algorithm (Algorithm 7), but instead of minimizing the cardinality of the set of consistent hypotheses (the 'version space'), the learner minimizes a subset of the domain (the 'danger zone').
14 Furthermore, the labels for most not-too-small subsets of the nodes reveal a lot of information about the correct labeling function -not just for a particular subset of nodes. These properties led us to code-name this construction while working on the paper as 'everything everywhere all at once' (in reference to a 2022 film of that name). This is in contrast to the 'minimal' function, where the information is concentrated entirely on the function path. The asymmetry between the 'minimal' class and the probabilistic class is similar to that between the binary and one-hot encodings in Section 2.3.1 above. 15 As well as depth-first order.
Concretely, at the beginning of the game the learner initializes a set S = {x 1 , x 2 , . . . , x tmax } consisting of the first t max = 2 Ω( √ d) instances in the sequence x selected by the adversary. This set represents the 'danger zone' -nodes in the beginning of the sequence that have not been labeled yet, that might be on-path, and that are not ancestors of a previously-labeled on-path node. 16 To predict a label for an instance x i , the learner selects a label ŷi such that if ŷi is wrong, the danger zone will shrink by at least 1/3. That is, for b ∈ {0, 1}, if the set S b of b-descendants of x i has cardinality |S b | ≥ |S|/3, the learner predicts ŷi = b. Then, if the adversary selects y i = 1b, that implies that all b-descendants of x i are off-path for the true labeling functions. Therefore, the learner removes all b-descendants of x i from the danger zone, and the new cardinality is |S \ S b | ≤ (2/3) • |S|. This guarantees that the learner can make at most O(log(t max )) = O √ d such mistakes before the danger zone is empty. 17If neither S 0 nor S 1 have cardinality at least |S|/3, the learner predicts ŷi = 0. If y i = 1 and x i is on-path for the true labeling function, then the learner updates the danger zone to be S 0 ∪ S 1 ,foot_7 again shrinking the danger zone by a factor of at most 2/3. Otherwise, if y i = 1 and x i is off-path, then it was an off-path node labeled 1 (which is rare), and the learner can afford to misclassify it (see Section 2.3.6).
this section cite: []

Section: Splitting Experts
The danger zone minimization strategy requires that the learner know whether the node u being classified is on-path or off-path for the true labeling function. However, if u appears in the sequence before some of its ancestors, the learner does not know this. To overcome this difficulty, the learner implements a variant of the standard multiplicative weights algorithm using splitting experts. This means that initially there is a single expert executing danger zone minimization. When a node u is reached for which danger zone minimization requires knowing whether u is on-path or off-path and that information is not yet evident, each expert is split into two experts, one of which continues the execution of danger zone minimization under the assumption that u is on-path, and the other under the opposite assumption. Thus, at each point in time, there exists precisely one expert for which all path-related assumptions are correct, and therefore that expert will make at most O √ d mistakes.
The multiplicative weights algorithm guarantees that the overall number of mistakes will be linear in the the number of mistakes of the best expert, i.e., O √ d .
this section cite: []

Section: Transition to Halving
The hypothesis class is engineered such that it satisfies the following property: there are at most 2 O( d) functions in the hypothesis class that agree with any set of t max = 2 Ω( √ d) labels, or that agree that a set of Θ √ d nodes are all off-path and labeled 1 (this follows from Lemma D.2).
Therefore, once the true labels for the first t max instances x 1 , x 2 , . . . , x tmax have been revealed, or once Θ √ d off-path labels of 1 have been revealed (whichever happens first), the learner can transition to halving: stop doing danger zone minimization, and instead predict the labels for the remaining nodes using the standard Halving algorithm (Algorithm 7) on the subset of the hypothesis class that survived. Halving on 2 O( d) functions is guaranteed to make at most O √ d mistakes (Fact E.1).
√
However, seeing as the learner lacks information on which nodes are off-path, it uses experts, and each expert maintains different path-related assumptions. Thus, each expert decides separately at which point to transition to Halving. The unique expert that makes only correct assumptions will transition 'at the right time'. That expert will make at most O √ d mistakes during danger zone minimization, and then at most O √ d additional mistakes during halving.
this section cite: []

Section: Some Intuition for the Quantity √ d
We briefly sketch where the quantity √ d arises from. This is a back-of-the-envelope calculation without proof, intended purely as an aid for intuition. Suppose we assigned off-path labels of 1 with probability 2 -k instead of 2 - √ d . Consider a sequence x 1 , . . . , x n of n = d/2k leaves. For any sequence of labels y 1 , . . . , y n ∈ {0, 1}, taking s = i∈[n] y i , there exist roughly
2 d • 2 -k s • 1 -2 -k n-s ≥ 2 d • 2 -k n ≫ 0
functions in the class for which these leaves are off-path and which agree with the labels y 1 , . . . , y n . Therefore, the adversary can force at least n = Ω(d/k) mistakes.
Similarly, for the sequence x 1 , . . . , x n consisting of all the nodes in the tree of depth at most k/2 in breadth-first order, the adversary can force a mistake on every on-path node while assigning a label of 0 to all off-path nodes, for a total of k/2 mistakes. This is true because for any assignment of on-path labels, the fraction of functions which agree with the on-path labels that assign a label of 0 to all off-path nodes is roughly 1 -2 -k 2 k/2 ≈ 1, so in particular for any labeling of the on-path nodes there exists a function in the class that agrees with that labeling and assigns 0 to all off-path nodes. Therefore, for any k, we obtain a lower bound of Ω d k + k on the number of mistakes. For any k,
d k + k ≥ √ d,
giving a lower bound of Ω √ d . Choosing k = √ d to minimize the lower bound will in fact yield a matching upper bound of O √ d , as we show in this paper. This completes our overview of the upper bound.
this section cite: []

Section: Directions for Future Work
Following are some interesting open questions:
1. Does there exist an efficient learning algorithm that achieves the O √ d upper bound of Theorem D.1? One needs to be careful about the definition of efficiency here, but one possible formalization is as follows. Does there exist a learning algorithm A and a sequence of classes H 1 , H 2 , . . . , such that for every d ∈ N:
• LD(H d ) = d, and • Given as input the index d and a sequence x 1 , . . . , x n , the algorithm A runs in time poly(d, n) and makes at most O √ d mistakes assuming the labels are realizable by H d .
2. Is there a tradeoff between the cardinality of the domain X and the upper bound on the number of mistakes? We used a domain of size roughly 2 d in order to obtain our upper bound of O √ d . Is it possible to get the same bound with a domain of size poly(d)?
3. Obtaining more precise asymptotics; for example, is there (an explicit) constant α > 0 such that the optimal transductive mistake bound is α + o(1) √ d?
this section cite: []

Section: Organization
Complete rigorous mathematical details are deferred to the appendices. Formal definitions appear in Section A. Formal statements and proofs for the lower bound and upper bound appear in Section B and Section D, respectively. Optimal sequence length is discussed in Section C.
this section cite: []

Section: Technical Appendices and Supplementary Material

this section cite: []

Section: A Preliminaries
A.1 Basic Notation Notation A.1. N = {1, 2, 3, . . .}, i.e., 0 / ∈ N. log(•) and ln(•) denote logarithm to base 2 and e, respectively. Notation A.2 (Sequences). Let X be a set and n, k ∈ N. For a sequence x = (x 1 , . . . , x n ) ∈ X n , we write x ≤k to denote the subsequence (x 1 , . . . , x k ). If k ≤ 0 then x ≤k denotes the empty sequence, which is also denoted by λ = X 0 . We use the notation X ≤n = ∪ n i=0 X i .
this section cite: []

Section: A.2 Standard Online Learning
Let X be a set, and let H ⊆ {0, 1} X be a collection of functions called a hypothesis class. A learner strategy or simply learner for the standard online learning game (Game 1) is a function
L : n-1 i=0 (X × {0, 1}) i × X → {0, 1},
where n ∈ N is the number of rounds in the game. The set of all such learner strategies is denoted L n .
An adversary strategy or simply adversary for the standard online learning game is a pair of functions
A instance : n-1 i=0 (X × {0, 1} × {0, 1}) i → X ,and
A label : n-1 i=1 (X × {0, 1} × {0, 1}) i × {0, 1} → {0, 1}.
The set of all such adversary strategies is denoted A n .
Semantically, the interpretation of these strategies is that in each round t ∈ [n] of Game 1, the adversary selects an instance x t = A instance (x 1 , ŷ1 , y 1 , . . . , x t-1 , ŷt-1 , y t-1 ) ∈ X , then the learner makes a prediction ŷt = L(x 1 , y 1 , . . . , x t-1 , y t-1 , x t ) ∈ {0, 1}, and finally, the adversary assigns a label y t = A label (x 1 , ŷ1 , y 1 , . . . , x t-1 , ŷt-1 , y t-1 , ŷt ) ∈ {0, 1}.
The adversary's function A label must satisfy realizability, meaning that there exists h ∈ H such that ∀t ∈ [n] : y t = h(x t ).
The number of mistakes in a game with n rounds and hypothesis class H between learner L and adversary A is M std (H, n, L, A) = |{t ∈ [n] : ŷt ̸ = y t }|.
this section cite: []

Section: A.3 Transductive Online Learning
Given X and H as in Section A.2, a learner strategy for the transductive online learning setting (Game 2) is a function
L : X n × n-1 i=0 {0, 1} i → {0, 1},
where n ∈ N is the number of rounds in the game. An adversary strategy consists of a sequence x ∈ X n and an adversary labeling strategy, which is a function
A : n-1 i=0 {0, 1} 2i × {0, 1} → {0, 1}.
The sets of all such learner and adversary strategies are denoted L n and A n respectively.
Semantically, the interpretation of these strategies is that at the start of Game 2, the adversary selects the sequence x. Then, in each round t ∈ [n], the learner makes a prediction ŷt = L(x, y 1 , . . . , y t-1 ) ∈ {0, 1}, and then the adversary assigns a label
y t = A(ŷ 1 , y 1 , . . . , ŷt-1 , y t-1 , ŷt ) ∈ {0, 1}.
Exactly as in Section A.2, the adversary's function A must satisfy realizability, namely, ∃h ∈ H ∀t ∈ [n] : y t = h(x t ), and the number of mistakes in a game with sequence length n and hypothesis class H between learner L and adversary A is M tr (H, n, L, A) = |{t ∈ [n] : ŷt ̸ = y t }|.
this section cite: []

Section: A.4 Mistake Bounds
In this paper, we study optimal mistake bounds, or the optimal number of mistakes, which is the value of Games 1 and 2. For M ∈ {M std , M tr }, the optimal number of mistakes in a game with hypothesis class H and sequence length n is,
M (H, n) = inf L∈Ln sup A∈An M (H, n, L, A).
The optimal number of mistakes for hypothesis class H is
M (H) = sup n∈N M (H, n).
Remark A.3. As is common in learning theory literature, in both Game 1 and Game 2, we take the sets L n and A n to be the sets of all (deterministic) functions. In this paper, we do not consider randomized strategies. By allowing arbitrary functions, we ignore issues relating to computability.
this section cite: []

Section: A.5 Trees
Definition A.4 (Notation for binary trees). Let d ∈ N ∪ {0}. A perfect binary tree of depth d is a collection of 2 d+1 -1 nodes, which we identify with the collection of binary strings
T d = {0, 1} k : k ∈ {0, 1, 2, . . . , d} .
The empty string, denoted λ = {0, 1} 0 , is a member of T d and is called the root of the tree. Every string u ∈ {0, 1} d is called a leaf. The depth of a node u ∈ T d , denoted |u|, is the length of u as a string, namely, the integer k such that u ∈ {0, 1} k .
For two nodes u, v ∈ T d , we say that u is a parent of v, and that v is a child of u,
if v = u • 0 or v = u • 1, where • denotes string concatenation. More fully, for b ∈ {0, 1}, we say that v is a b-child of u if v = u • b.
Recursively, we define that u is an ancestor of v and that v is a descendant of u, and write u ≼ v, if one of the following holds:
• u = v, or • ∃w ∈ T d ∃b ∈ {0, 1} : (u ≼ w) ∧ (w • b = v).
For b ∈ {0, 1}, we say that v is a b-descendant of u, denoted u ≼ b v, if v is a descendant of the b-child of u.
A function f : T d → {0, 1} specifies a particular root-to-leaf path in the tree T d (see Figure 1). The on-path nodes for f are the set of d + 1 nodes on that root-to-leaf path, as in the following definition.
Definition A.5 (Paths in a binary tree). Let d, k ∈ N, k ≤ d. Let u ∈ {0, 1} k be a node in T d .
The path to u is the unique sequence path(u) = (u 0 , u 1 , u 2 , . . . , u k ) such that u 0 = λ is the root, u k = u, and u i is a child of u i-1 for all i ∈ [k].
Let f : T d → {0, 1} be a function. The path of f is the unique sequence path(f ) = (u 0 , u 1 , u 2 , . . . , u d ) such that u 0 = λ is the root, and for each i ∈
[d], u i = u i-1 • f (u i-1 ).
Namely, u i is the f (u i-1 )-child of u i-1 .
For a node v ∈ T d and a function f :
T d → {0, 1}, we write v ∈ path(f ) if path(f ) = (u 0 , . . . , u d )
and there exists i ∈ {0, . . . , d} such that u i = v. Otherwise, we write v / ∈ path(f ).
For a node v ∈ T d and a set of functions F ⊆ {0, 1} T d , we write v ∈ path(F ) if ∀f ∈ F : v ∈ path(f ).
Otherwise, we write u / ∈ path(F ).
this section cite: []

Section: A.6 Littlestone Dimension
Definition A.6 (Littlestone, 1987). Let X be a set, let H ⊆ {0, 1} X , and let d ∈ N ∪ {0}. We say that H shatters the binary tree T d if there exists a mapping T d → X given by u → x u such that for every u ∈ {0, 1} d+1 there exists h u ∈ H such that
∀i ∈ [d + 1] : h(x u ≤i-1 ) = u i .
The Littlestone dimension of H, denoted LD(H), is the supremum over all d ∈ N such that there exists a Littlestone tree of depth d -1 that is shattered by H.
Note that by defining the Littlestone dimension this way, every class with Littlestone dimension d ∈ N contains at least 2 d functions. Theorem A.7 (Littlestone, 1987). Let X be a set and let H ⊆ {0, 1} X such that d = LD(H) < ∞.
0 1 0 1 0 1 0 1 0 1 1 0 1 0 x λ x 0 x 1 x 00 x 01 x 10 x 11 ∃ h ∈ H : h(x λ ) = 1 h(x 1 ) = 0 h(x 10 ) = 1
Then there exists a strategy for the learner that guarantees that the learner will make at most d mistakes in the standard (non-transductive) online learning setting, regardless of the adversary's strategy and of the number n of instances to be labeled. Furthermore, there exists an adversary that forces every learner to make at least min {n, d} mistakes.
this section cite: ['b18', 'b18']

Section: B Lower Bound

this section cite: []

Section: B.1 Statement
Our Ω √ d lower bound states the following.
this section cite: []

Section: Theorem B.1 (Lower bound).
There exists a constant d 0 ≥ 0 as follows. Let d ∈ N, d ≥ d 0 , let X be a set, and let H ⊆ {0, 1} X be a hypothesis class with LD(H) = d. Then there exist a sequence
x ∈ X n of length n = O d • 2 √ d and an adversary A that always selects the sequence x and uses a simple adaptive labeling strategy (as in Algorithm 1), such that for every learning rule L, M tr (H, n, L, A) ≥ √ d/10.
Furthermore, for every integer n ∈ N,
M tr (H, n) ≥ min √ d/10, ⌊log(n + 1)⌋ .(4)
Remark B.2. The assumption LD(H) = d implies that for all k ∈ [d], H shatters a Littlestone tree of depth k. Thus, the lower bound of Eq. ( 3) in Theorem B.1 immediately implies that for every k ∈ [d] there exists a sequence x (k) ∈ X n k of length n k = O k • 2 √ k such that the adversary A k that presents the sequence x (k) and assigns labels using the simple labeling strategy of Algorithm 1 ensures that for every learner L,
M tr (H, n k , L, A k ) ≥ √ k/10.
See Section 2.2 for a general overview of Theorem B.1 and the main proof ideas. In the following subsections we prove Theorem B.1. Algorithm 1 gives an explicit construction of the adversary that witnesses the lower bound, using Algorithm 2 as a subroutine. We start with presenting some initial observations about the behavior of these algorithms in Section B.2.
Assumptions:
• d ∈ N, ε = 2 - √ d/2 .
• T = T d is a perfect binary tree of depth d.
• H ⊆ {0, 1} T is a class that shatters T .
TRANSDUCTIVEADVERSARY (H):
(x 1 , x 2 , . . . , x n ) ← CONSTRUCTSEQUENCE (H) ▷ See Algorithm 2. send (x 1 , x 2 , . . . , x n ) to learner H 0 ← H for t ∈ [n]: receive ŷt from learner r t ← |{h ∈ H t-1 : h(x t ) = 1}| |H t-1 | y maj ← 1(r t ≥ 1/2) y t ← y maj r t / ∈ [ε, 1 -ε] 1 -ŷt otherwise send y t to learner H t ← {h ∈ H t-1 : h(x t ) = y t }
Algorithm 1: The strategy for the adversary that achieves the lower bound in Theorem B.1. Note that while the construction of the sequence x is not entirely trivial, the adversary's strategy for labeling this sequence is very simple.
this section cite: []

Section: B.2 Analysis of the Adversary
Claim B.3. Let d ∈ N, let M = √ d/10, and let H ⊆ {0, 1} T d be a hypothesis class. Consider an execution of CONSTRUCTSEQUENCE (H) as in Algorithm 2 that produces a sequence x 1 , x 2 , . . . , x n . Then:
Assumptions:
• d ∈ N, M = √ d/10, ε = 2 - √ d/2 .
• T = T d is a perfect binary tree of depth d.
• λ, the empty string, is the root of T .
• H ⊆ {0, 1} T is a class that shatters T . CONSTRUCTSEQUENCE (H):
H λ ← H H 0 ← {H λ } ▷ A set of classes indexed by bit strings.
Q ← {λ} ▷ A set of nodes to be processed.
t ← 0 while |Q| > 0: t ← t + 1
x t ← arbitrary element from Q ▷ Pop an arbitrary element from Q and add it to the output sequence.
Q ← Q \ {x t } H t ← ∅ for H b ∈ H t-1 : r ← |{h ∈ H b : h(x t ) = 1}| |H b | Y ← {0, 1} r ∈ [ε, 1 -ε] ∧ |b| < M {1(r ≥ 1/2)} otherwise
▷ Adversary will force mistakes on the first M balanced nodes. for y ∈ Y:
b ′ ← b |Y| = 1 b • y |Y| = 2
▷ Restrict class to agree with y. If splitting the class in two to force a mistake then create new indices.
H b ′ ← {h ∈ H b : h(x t ) = y} H t ← H t ∪ {H b ′ } if x t ∈ path(H b ′ ) ∧ |x t | < d: ▷ If x t is on-path for H b ′
and it has a y-child, add that child to Q.
Q ← Q ∪ {x t • y} return (x 1 , x 2 , . . . , x t )
Algorithm 2: A subroutine of Algorithm 1 for selecting the sequence x.
(a) For all i ∈ [n], path(x i ) is a subsequence of x 0 , x 1 , . . . , x i .
(b) The length n of the sequence satisfies n < n d , where
n d = (d + 1) • 2 M +1 . Proof. (a) Fix i ∈ [n]. It suffices to show that for all u ∈ T d , if u ≼ x i then u ∈ (x 1 , x 2 , . . . , x i ).
Proceed by induction on i. For the base case i = 1, the claim holds because x 1 = λ.
For the induction step, assume the claim holds for i ∈ [n -1]. Let u ≼ x i+1 , we prove that u ∈ (x 1 , x 2 , . . . , x i+1 ). Assume x i+1 ̸ = λ (otherwise, there is nothing to prove).
Because x i+1 appears in the sequence x, it must have been added to Q before it was added to x. The only place where items that are not λ are added to Q is in the line Q ← Q ∪ {x t • y}. Namely, there exist an index j ∈ [i] and a bit y ∈ {0, 1} such that x i+1 = x j • y (note that j < i + 1 because x j was added to the sequence before x i+1 ). If x j = u we are done. Otherwise, note that x j is the parent of x i+1 , and therefore u ≼ x j . By the induction hypothesis, u ∈ (x 1 , x 2 , . . . , x j ). This concludes the proof.
(b) Items are added to the sequence x only if they were previously added to Q. By induction on i ∈ [n], for each x i in the sequence, there is at most one iteration of the "while |Q| > 0" loop in which x i is added to Q. The base case i = 1 holds because x 1 = λ is the root, which is added to Q before the while loop, and λ is never added to Q within that loop because the line "Q ← Q ∪ {x t • y}" can only add non-empty bit strings. For the induction step, if the claim holds for all natural numbers j such that 1 ≤ j < i ≤ n then it holds for i. Indeed, for i ≥ 2, x i can be added to Q only via the line "Q ← Q ∪ {x t • y}", and only in the iteration of the while loop where x t is the parent of x i in the tree T d . In that iteration, the parent x t of x i is popped from Q, which implies that x t was added to Q in some previous iteration of the while loop (t < i), and is no longer in Q after being popped. By the induction hypothesis, x t will never be added to Q again, and therefore in all subsequent iterations of the while loop x t will not be the parent of x i , so x i cannot be added to Q in subsequent iterations via the line "Q ← Q ∪ {x t • y}".
Furthermore, if a node x i is added to Q in some iteration of the while loop, then it remains in Q for the duration of that iteration. So for all i ∈ {2, 3, . . . , n}, there is precisely one execution of the line "Q ← Q ∪ {x t • y}" that adds x i to Q. Namely, there is precisely one point in time during the execution of Algorithm 2 in which x i = x t • y, x i / ∈ Q, and the line "Q ← Q ∪ {x t • y}" is executed resulting in x i ∈ Q.
Consider a function f that maps i ∈ {2, 3, . . . , n} to the value of the index b ′ during the unique execution of the line "Q ← Q ∪ {x t • y}" that adds x i to Q. Namely, if b ′ had some value β when x i was added to Q, then f (i) = β.
Notice that "Q ← Q∪{x t •y}" is executed only if the condition x t ∈ path(H b ′ ) is satisfied in the previous line. Furthermore, the line "H b ′ ← {h ∈ H b : h(x t ) = y}" ensures that the node x i = x t • y being added to Q satisfies x t • y ∈ path(H b ′ ), namely ∀h ∈ H b ′ : x i ∈ path(h).
Consequently, x i ∈ path(G) for any class G that is a subset of H b ′ ; in particular, because the only way that H b ′ might be modified later during the execution of Algorithm 2 is by removing elements, it follows that x i ∈ path(H b ′ ) when the line "Q ← Q ∪ {x t • y}" is executed and in all subsequent times. However, |path(G)| = d + 1 for any class G ⊆ {0, 1} T d . This implies that f maps at most (d + 1) nodes to each bit string. In other words, for any bit string b, the size of the preimage satisfies |f
-1 (b)| ≤ d + 1. The condition "|b| < M " in Algorithm 2 ensures that |b ′ | ≤ M , namely, b ′ ∈ {0, 1} k for k ∈ {0, 1, 2, . . . , M }. Thus, n = 1 + |{2, 3, . . . , n}| = 1 + b∈{0,1} k k∈{0,...,M } |{i ∈ {2, 3, . . . , n} : f (i) = b}| = 1 + b∈{0,1} k k∈{0,...,M } |f -1 (b)| ≤ 1 + b∈{0,1} k k∈{0,...,M } (d + 1) ≤ 1 + (d + 1) • (2 M +1 -1). < (d + 1) • 2 M +1 , as desired.
Claim B.4. Let d ∈ N, let M = √ d/10, and let H ⊆ {0, 1} T d be a hypothesis class. Consider an execution of TRANSDUCTIVEADVERSARY (H) as in Algorithm 1. Let H 0 , H 1 , . . . , H n be the sequence of hypothesis classes created by TRANSDUCTIVEADVERSARY, let
S = t ∈ [n] : r t ∈ [ε, 1 -ε]
be the set of indices where TRANSDUCTIVEADVERSARY forces a mistake, and let H 0 , H 1 , . . . , H n be the sequence of collections created by the subroutine CONSTRUCTSEQUENCE (Algorithm 2). If |S| ≤ M then ∀t ∈ {0, 1, . . . , n} : H t ∈ H t .
Proof. Proceed by induction on t ∈ {0, 1, . . . , n}. The base case t = 0 is satisfied, because H 0 = H ∈ {H} = H 0 . For the induction step, assume that H i-1 ∈ H i-1 for some i ∈ [n]. We prove that H i ∈ H i .
Let y i be the label assigned to x i by TRANSDUCTIVEADVERSARY. Then
H i = {h ∈ H i-1 : h(x i ) = y i }.
Consider the iteration of the while loop in CONSTRUCTSEQUENCE that starts with t ← i. By the induction hypothesis, H i-1 ∈ H i-1 . Therefore, in this iteration of the while loop, there will be an iteration of the "for H b ∈ H t-1 " loop where H b = H i-1 . In that iteration, y i ∈ Y by construction of y i and Y. Therefore, in the iteration of the "for y ∈ Y" loop in which y = y i ,
H b ′ = {h ∈ H b : h(x t ) = y} = {h ∈ H i-1 : h(x i ) = y i } = H i .
The class H b ′ is then added to H i = H t in the line "H t ← H t ∪ {H b ′ }". Furthermore, no class is ever removed from H t . So H i ∈ H i , as desired.
Claim B.5. Let d ∈ N, let M = √ d/10, and let H ⊆ {0, 1} T d be a hypothesis class. Consider an execution of TRANSDUCTIVEADVERSARY (H) as in Algorithm 1 where the adversary constructs a sequence of nodes x 1 , x 2 , . . . , x n ∈ T d and a sequence of classes H 0 , H 1 , . . . , H n ⊆ {0, 1} T d . Let
S = t ∈ [n] : r t ∈ [ε, 1 -ε]
be the set of indices where TRANSDUCTIVEADVERSARY forces a mistake, and assume that |S| ≤ M . Then for all k ∈ {0, 1, . . . , d} there exists i ∈ [n] such that 1. |x i | = k, and 2. x i ∈ path(H i-1 ), Proof. Proceed by induction on k. For the base case k = 0, notice that x 1 = λ, |λ| = 0, and λ ∈ path(H -1 ).
For the induction step, assume the claim holds for some k ∈ {0, 1, . . . , d -1}, and take i k ∈ [n] such that |x i k | = k and x i k ∈ path(H i k -1 ); we prove that the claim holds for k + 1 as well.
Consider the iteration of the while loop in CONSTRUCTSEQUENCE in which x i k is added to the sequence (i.e., the iteration starting with t ← i k ). By Claim B.4 and the assumption |S| ≤ M , H i k -1 ∈ H i k -1 . Hence, within this iteration of the while loop, there is an iteration of the "for H b ∈ H t-1 " loop such that H b = H i k -1 . By construction, the set Y always contains the label predicted by the adversary, so y i k ∈ Y. Consider the iteration of the "for y ∈ Y" loop such that y = y i k . By the induction hypothesis, x i ∈ path(H i k -1 ), and since H b ′ ⊆ H b = H i k -1 , it follows that x i k ∈ path(H b ′ ). Seeing as |x i k | < d, in the last line of this iteration of the "for y ∈ Y" loop, the node x i k+1 := x i k • y i k is added to Q. This guarantees that x i k+1 will eventually be popped from Q and added to the sequence returned by CONSTRUCTSEQUENCE. Once a node has been added to the sequence, it is never removed.
Notice that |x i k+1 | = |x i k | + 1 = k + 1, satisfying Item 1. Therefore, it remains to show Item 2, namely, to show that x i k+1 ∈ path H i k+1 -1 . Indeed, by the induction hypothesis, x i ∈ path(H i k -1 ), and in the iteration of the "for y ∈ Y" discussed above, H b = H i k -1 , H b ′ = H i k , and
H b ′ = {h ∈ H b : h(x i k ) = y i k }. Hence, ∀h ∈ H i k : x i k ∈ path(h) ∧ h(x i k ) = y i k .
Seeing as
x i k+1 = x i k • y i k This implies that ∀h ∈ H i k : x i k+1 ∈ path(h).
Item 2 follows from the inclusion H i k+1 -1 ⊆ H i k .
this section cite: []

Section: B.3 Proof
Finally, we complete the proof of the lower bound.
Proof of Theorem B.1. Fix d 0 = 800 and assume d ≥ d 0 . Seeing as LD(H) = d, H shatters the tree T d . By replacing H with a suitable subset of H of cardinality 2 d+1 , renaming the elements in the domain of H to nodes of T d , and restricting the domain of each function in H to T d , assume without loss of generality that H ⊆ {0, 1} T d , |H| = 2 d+1 , and H shatters T d .
Consider the loop "for t ∈ [n]" in Algorithm 1, and let
S = {s 1 , s 2 , . . . , s m } = t ∈ [n] : r t ∈ [ε, 1 -ε]
be the set of indices where the adversary forces a mistake, such that the learner makes at least m = |S| mistakes. Let M = √ d/10, and assume for contradiction that m ≤ M .
By Claim B.5, there exists t ∈ [n] such that |x t | = d (i.e., x t is a leaf in T d ) and x t ∈ path(H t-1 ), namely, ∀h ∈ H t-1 : x t ∈ path(h).
Seeing as x t is a leaf, ∀h ∈ H t-1 : path(x t ) = path(h).
(5) By construction, H t ⊆ h ∈ H : (∀i ∈ [t] : h(x i ) = y i ) ,
and H t is not empty. Fix some h * ∈ H t ⊆ H t-1 . By Item (a) in Claim B.3, path(x t ) = path(h * ) is a subsequence of x 1 , x 2 , . . . , x t , so ∀h ∈ H t ∀x ∈ path(h * ) : h(x) = h * (x).
Seeing as H shatters T d and |H| = 2 k+1 , if two functions h, h * ∈ H agree on the labels for all nodes in path(h * ), then h = h * . We conclude that H t = {h * } and |H t | = 1.
Consider the loop "for t ∈ [n]" in Algorithm 1. For each t ∈ [n],
|H t | ≥ ε • |H t-1 | t ∈ S (1 -ε) • |H t-1 | t / ∈ S.
Hence,
1 = |H t | ≥ ε m • (1 -ε) n-m • |H 0 | = ε m • (1 -ε) n-m • 2 d+1 ≥ ε m • (1 -ε) n • 2 d+1 ≥ ε m • (1 -ε) n d • 2 d+1 (by Item (b) in Claim B.3.) ≥ ε m • 2 d = 2 -m √ d/2+d ,(6)
where the final line holds because ε = 2 - √ d/2 , n d = (d + 1) • 2 √ d/10+1 , and
(1 -ε) n d = 1 -2 - √ d/2 (d+1)•2 √ d/10+1 ≥ 1 2
for our choice of d ≥ 800. Rearranging Eq. ( 6) yields
2 √ d ≤ m.
This is a contradiction to the assumption m ≤ M = √ d/10. We conclude that an adversary A following Algorithm 1 satisfies
inf L∈Ln M tr (H, n, L, A) ≥ m > M = √ d/10,(7)
as desired.
To establish the "furthermore" part of the theorem, fix a length n ∈ N. Let k be the largest integer such that 2 ⌈ √ k/10⌉ ≤ n + 1 and k ≤ d. By Eq. ( 7), there exists some sequence on which the adversary can force every learning rule to make at least √ k/10 mistakes. By Theorem C.2, this implies that there exists a sequence of length 2 ⌈ √ k/10⌉ -1 ≤ n on which the adversary can force every learning rule to make at least √ k/10 = min √ d/10 , ⌊log(n + 1)⌋ mistakes. Namely,
M tr (H, n) ≥ min √ d/10 , ⌊log(n + 1)⌋ ,
as in Eq. ( 4).
this section cite: []

Section: C Sequence Length
In this section, we show that if there exists a sequence on which the adversary can force M mistakes, then a sequence of length 2 M -1 is sufficient, and this upper bound is tight for some classes. 19Definition C.1 (Minimal sequence). Let X be a set, let H ⊆ {0, 1} X be a class, and let M ∈ N.
The minimal sequence length for forcing M mistakes for the class H, denoted MinLen(H, M ) is MinLen(H, M ) = inf {n ∈ N : (∃x ∈ X n : M tr (H, x) ≥ M )}.
In words, MinLen(H, M ) is the smallest integer n for which there exists a sequence of length n on which the adversary can force at least n mistakes; if no such sequence exists, then MinLen(H, M ) = ∞.
Theorem C.2 (Minimal sequence bound). Let X be a set, and fix M ∈ N. Then for any class H ⊆ {0, 1} X , if MinLen(H, M ) < ∞ then MinLen(H, M ) ≤ 2 M -1.
Furthermore, there exists a class H ⊆ {0, 1} X for which MinLen(H, M ) = 2 M -1.
Theorem C.2 is a corollary of the tree rank characterization of M tr from Ben-David et al. (1997).
For completeness, we present a direct proof of Theorem C.2 that does not directly invoke that characterization. Roughly, given an adversary A 0 that forces every learner to make at least M mistakes on a (possibly long) sequence x, we apply two modifications to obtain new adversaries
A 0 ⇝ A 1 ⇝ A 2 .
A 1 forces M mistakes and has a specific structure that we call 'rigidity', but it still uses the same (possibly long) sequence x. Capitalizing on the rigid structure, A 2 selects a subsequence of x of length at most 2 M -1, and forces M mistakes on that subsequence.
this section cite: ['b2']

Section: C.1 Rigid Adversary
Definition C.3 (Rigid adversary). Let n ∈ N, let X be a set, and let
A : n-1 k=0 {0, 1} 2k × {0, 1} → {0, 1}
be an adversary strategy for some fixed sequence x ∈ X n . We say that A is rigid if there exists a function
f : n-1 k=0 {0, 1} k → {0, 1, ⋆}
such that for all k ∈ {0, 1, . . . , n -1} and all y, ŷ ∈ {0, 1} k ,
A(ŷ 1 , y 1 , . . . , ŷk , y k , ŷk+1 ) = f (y 1 , . . . , y k ) f (y 1 , . . . , y k ) ∈ {0, 1} 1 -ŷk+1 f (y 1 , . . . , y k ) = ⋆ .
Note that if an adversary is rigid, then the function f that witnesses this is uniquely determined. Claim C.4 (Rigid adversary exists). Let n, M ∈ N, let X be a set, let x ∈ X n , and let H ⊆ {0, 1} X be a class. Let A be an adversary strategy that forces every learner to make at least M mistakes on x.
Then there exists an adversary strategy A * such that:
1. A * forces every learner to make at least M mistakes on x and A * is rigid.
2. Let f be the function that witnesses the rigidity of A * . Then for every y ∈ {0, 1} n , the sequence
f (y ≤0 ), f (y ≤1 ), f (y ≤2 ), . . . , f (y),
has at least M members equal to ⋆.
Proof of Claim C.4. For Item 1, consider the adversary strategy A * that simulates an execution of A, as in Algorithm 3. In broad strokes, A * functions as a middle-man between the learner and A. As the learner makes a sequence of predictions ŷ ∈ {0, 1} n , the adversary A * generates a sequence of (possibly different) predictions ỹ ∈ {0, 1} n , and sends those to the adversary A. Adversary A sees only the predictions ỹ, and assigns labels y ∈ {0, 1} n , which are relayed back to the learner by A * with no modifications.
First, observe that A * satisfies the realizability requirement. Indeed, A * simulates an execution of A such that the sequence of labels y 1 , . . . , y n sent by A * to the learner is exactly the sequence of labels selected by A. Seeing as A is realizable, every sequence of labels selected by A is realizable, and therefore every sequence of labels selected by A * must be realizable as well.
Second, observe that A * forces every leaner to make at least M mistakes. To see this, notice that in Algorithm 3,
t∈[n] 1(ỹ t ̸ = y t ) ≥ M.(8)
Indeed, A forces every learner to make at least M mistakes, and in particular this applies to a learner that makes predictions ỹ as in the simulation. Furthermore, observe that A * only alters the predictions it receives from the learner in cases when it selects a label that is accepted by A, namely, ∀t ∈ [n] : ỹt ̸ = ŷt =⇒ ỹt = y t .
Therefore, if E = {t ∈ [n] : ỹt = ŷt }, then
t∈[n] 1(ỹ t ̸ = y t ) = t∈E 1(ỹ t ̸ = y t ) + t∈[n]\E 1(ỹ t ̸ = y t ) = t∈E 1(ỹ t ̸ = y t ) + 0 (By Eq. (9)) = t∈E 1(ŷ t ̸ = y t ) (Defintion of E)
Assumptions:
• n ∈ N, X is a set, x ∈ X n is a fixed sequence of instances.
• A :
n-1 k=0 {0, 1} 2k × {0, 1} → {0, 1} is an adversary labeling strategy for x.
this section cite: []

Section: RIGIDADVERSARY:
send x 1 , . . . , x n to the learner for t = 1, 2, . . . , n:
receive prediction ŷt from learner if A(ỹ 1 , y 1 , . . . , ỹt-1 , y t-1 , 0) = 0: ỹt ← 0 else if A(ỹ 1 , y 1 , . . . , ỹt-1 , y t-1 , 1) = 1: ỹt ← 1 else: ỹt ← ŷt send prediction ỹt to A receive label y t from A send label y t to learner
Algorithm 3: Construction of a rigid adversary, by simulating a given adversary A.
≤ t∈[n] 1(ŷ t ̸ = y t ).(10)
Combining Eqs. ( 8) and (10) implies that A forces at least M mistakes.
Third, we show that A * is rigid. We claim that there exists a function g : {0, 1} ≤n-1 → {0, 1} ≤n-1 such that for every t ∈ {0, 1, 2, . . . , n -1}, (ỹ 1 , . . . , ỹt ) = g(y 1 , . . . , y t ).
this section cite: []

Section: Proceed by induction on t.
For the base case t = 0 there is nothing to prove. For the induction step, we assume the claim holds for some t = k < n -1, and show that it holds for t = k + 1. From Algorithm 3, ỹk+1 satisfies
ỹk+1 = 0 A(ỹ 1 , y 1 , . . . , ỹk , y k , 0) = 0 1 A(ỹ 1 , y 1 , . . . , ỹk , y k , 0) = A(ỹ 1 , y 1 , . . . , ỹk , y k , 1) = 1 1 -y k+1 otherwise . (11
)
The first two cases in Eq. ( 11) are immediate from Algorithm 3, and the remaining case occurs when A forces a mistake at time k + 1, namely, when A selects y k+1 = 1 -ỹk+1 . Thus, ỹk+1 is a function of y ≤k+1 and ỹ≤k . By the induction hypothesis, ỹ≤k = g(y ≤k ), so ỹk+1 is simply a function of y ≤k+1 . This establishes the existence of the desired function g.
Hence, A * is rigid, as witnessed by the function
f (y 1 , . . . , y k ) = 0 A(ỹ 1 , y 1 , . . . , ỹk , y k , 0) = 0 1 A(ỹ 1 , y 1 , . . . , ỹk , y k , 0) = A(ỹ 1 , y 1 , . . . , ỹk , y k , 1) = 1 ⋆ otherwise ,
where f is a well-defined function because ỹ≤k = g(y ≤k ).
We have seen that A * is a valid (realizable) adversary that forces every learner to make at least M mistakes, and it is rigid. This concludes the proof of Item 1.
Finally, For Item 2, note that ỹt ̸ = y t only if A forces a mistake at time t in the sense that A selects y t = 1b for any prediction b ∈ {0, 1} provided at time t. If A forces a mistake at time t, then A * forces a mistake at time t as well. Therefore, if ỹt ̸ = y t , then f (y <t ) = ⋆, namely, ỹt makes mistakes only when the value of f is ⋆. By Eq. ( 8), ỹt makes at least M mistakes throughout the game, so there must be at least M rounds where f outputs ⋆, as desired.
this section cite: []

Section: C.2 Essential Indices
Definition C.5. Let n, M ∈ N, let X be a set, let x ∈ X n , and let H ⊆ {0, 1} X be a class. Let A be a rigid adversary strategy witnessed by function f . We say that an index t ∈ [n] is essential for A for forcing M mistakes on x if there exists a sequence y ∈ {0, 1} t-1 such that f (y) = ⋆ and the sequence f (y ≤0 ), f (y ≤1 ), f (y ≤2 ), . . . , f (y ≤t-1 )
contains at most M -1 members equal to ⋆.
Claim C.6. Let n, M ∈ N, let X be a set, let x ∈ X n , and let H ⊆ {0, 1} X be a class. Let A be a rigid adversary strategy. Then [n] contains at most 2 M -1 indices that are essential for A for forcing M mistakes on x.
Proof. For each essential index t ∈ [n], there exists a label sequence y ∈ {0, 1} t-1 that witnesses that t is essential, as in Definition C.5. Each label sequence y is a witness for at most one index (the index |y| + 1), so it suffices to show that the set Y ⊆ {0, 1} ≤n-1 of all witness label sequences is of cardinality at most 2 M -1.
Think of Y as a collection of nodes in the binary tree T n-1 (Definition A.4). By Definition C.5, if y ∈ Y , then the collection of all ancestors of y in Y has cardinality
y ≤i : i ∈ {0, 1, 2, . . . , |y| -1} ∩ Y ≤ M -1.
Namely, Y is a subtree of depth at most d = M -1 in the binary tree T n-1 . 20 Hence, the number of nodes in Y is at most 2 d+1 -1 = 2 M -1, as desired.
this section cite: []

Section: C.3 Proof
Proof of Theorem C.2. If MinLen(H, M ) < ∞, then there exist a sequence x ∈ X n , and an adversary A 0 that forces every learner to make at least M mistakes on x. By Claim C.4, there exists a rigid adversary A 1 that causes every learner to make at least M mistakes on x,foot_10 and also satisfies Item 2 in Claim C.4. Let f be the function that witnesses the rigidity of A 1 . By Claim C.6, the set I ⊆ [n] of indices that are essential for A 1 for forcing M mistakes on x has cardinality k = |I| ≤ 2 M -1.
Algorithm 4 defines a new adversary, A 2 , which forces every learner to make at least M mistakes on a sequence of length k. A 2 is realizable, because A 1 is realizable. 22To see that adversary A 2 forces every learner to make at least M mistakes, let y 1 , . . . , y n be the sequence of labels assigned by A 2 . Seeing as A 2 assigns the same labels as A 1 , and A 1 satisfies Item 2 in Claim C.4, it follows that there are at least M indices j ∈ [n] such that f (y ≤j-1 ) = ⋆. Fix J ⊆ [n] to be the first M such indices. Then J ⊆ I, namely, all the indices in J are essential for A 1 for forcing M mistakes on x (Definition C.5).
Therefore, for each j ∈ J, A 2 includes the instance x j in the sequence of length k sent to the learner. Then, in round j of the n rounds simulated by A 2 :
• The leaner makes a prediction ŷj ∈ {0, 1} corresponding to instance x j .
• Adversary A 2 sends prediction ŷj to adversary A 1 . Because f (y ≤j-1 ) = ⋆, adversary A 1 assigns the label y j = 1ŷj . Adversary A 2 then sends that label y j to the learner. So the learner makes a mistake on x j .
Hence, the learner makes at least |J| = M mistakes, as desired.
Assumptions:
• n, M ∈ N, X is a set, x ∈ X n is a fixed sequence of instances.
• A 1 :
n-1 k=0 {0, 1} 2k × {0, 1} → {0, 1} is a rigid adversary labeling strategy for x that forces every learner to make at least M mistakes on the sequence x, and satisfies Items 1 and 2 in Claim C.4.
• I = {i 1 , i 2 , . . . , i k } ⊆ [n] is the set of indices that are essential for A for forcing M mistakes on x, and
i 1 ≤ i 2 ≤ • • • ≤ i k . By Claim C.6, k ≤ 2 M -1. MINIMALADVERSARY: send x i1 , x i2 , . . . , x i k to the learner for t = 1, 2, . . . , n: if t ∈ I: receive prediction ŷt from learner send prediction ŷt to A 1 receive label y t from A 1 send label y t to learner else: send prediction ŷt = 0 to A 1 receive label y t from A 1
Algorithm 4: Construction of an adversary that forces M mistakes using a sequence x of length at most 2 M -1. In the proof of Theorem C.2, this adversary is A 2 . Internally, it simulates a rigid adversary A 1 .
this section cite: []

Section: D Upper Bound

this section cite: []

Section: D.1 Statement
The following result states that the lower bound of Theorem B.1 is tight for some classes. Theorem D.1 (Upper bound, and separation between standard and transductive online learning).
For every integer d ≥ 43, there exists a hypothesis class H ⊆ {0, 1} X with a domain X of size |X | = 2 d -1 such that LD(H) = d and the following two conditions hold for all n ∈ N:
1. M tr (H, n) ≤ 48 • √ d. 2. M std (H, n) = min {n, d}.
this section cite: []

Section: D.2 Hypothesis Class
In this section we construct the hypothesis class for Theorem D.1. Lemma D.2. Let d ∈ N, d ≥ 42. Let T d be a perfect binary tree of depth d, as in Definition A.4. Then there exists a collection of functions H ⊆ {0, 1} T d such that LD(H) = d + 1 and the following two conditions hold for all H ⊆ H and all X ⊆ T d :
1. If ∀h ∈ H ∀x ∈ X : x / ∈ path(h) ∧ h(x) = 0, then min {|H|, |X|} < 2 2 √ d . 2. If ∀h ∈ H ∀x ∈ X : x / ∈ path(h) ∧ h(x) = 1, then |H| < 2 2 √ d or |X| < 3 √ d.
The proof employs the probabilistic method, showing that a hypothesis class sampled randomly from a suitable distribution has the desired properties with very high probability.
Proof. Let P be a probability distribution over hypothesis classes.
Formally, P ∈ ∆ ({0, 1} T d ) 2 d+1
is a distribution over vectors of hypotheses. Each vector H ∈ supp(P) consists
of 2 d+1 hypotheses, H = (h b ) b∈{0,1} d+1 ,
where for each b ∈ {0, 1} d+1 , hypothesis h b is a function h b : T d → {0, 1} sampled independently as follows:
• For each i ∈ [d] ∪ {0}: h b (b ≤i ) = b i+1 . (In particular, with probability 1, path(h b ) = (b ≤0 , b ≤1 , . . . , b ≤d )
, each entry in the vector H is unique, and H shatters T d .)
• For each x ∈ T d \path(h b ), the bit h b (x) ∈ {0, 1} is sampled Ber 2 - √ d independently of all other bits in H, i.e., P[h b (x) = 1] = P[h b (x) = 1 | {h b ′ } b ′ ̸ =b , {h b (x ′ )} x ′ ̸ =x ] = 2 - √ d .
In words, for all nodes on the path in the tree corresponding to b, the function h b assigns a label according to b, and for all other nodes, h b assigns a label of 1 with probability 2 - √ d , and a label of 0 otherwise. In particular, the collection H Littlestone-shatters the tree
T d . Fix B ⊆ {0, 1} d+1 and X ⊆ T d , and let E(B, X, y) denote the event {∀b ∈ B ∀x ∈ X : x / ∈ path(h b ) ∧ h b (x) = y}.(12)
Seeing as each off-path label h b (x) ∈ {0, 1} is sampled independently,
P H∼P [E(B, X, 0)] = (b,x)∈B×X P H∼P [x / ∈ path(h b ) ∧ h b (x) = 0] ≤ (1 -2 - √ d ) |B×X| .(13)
Hence,
P H∼P ∃B ⊆ {0, 1} d+1 ∃ X ⊆ T d : E(B, X, 0) ∧ min {|B|, |X|} ≥ 2 2 √ d = P H∼P ∃B ⊆ {0, 1} d+1 ∃ X ⊆ T d : E(B, X, 0) ∧ |B| = |X| = 2 2 √ d ≤ |{0, 1} d+1 | 2 2 √ d |T d | 2 2 √ d (1 -2 - √ d ) 2 4 √ d (union bound, Eq. (13)) < 2 d+1 2 2 √ d + 1 2 • (1 -2 - √ d ) 2 4 √ d (2 2 √ d
may not be an integer; formally, we use the generalized binomial coefficient, or simply skip to the next line)
< 2 2•(d+1)• 2 2 √ d +1 • e -2 -√ d •2 4 √ d ( n k < n k for k ≥ e; 1 + x ≤ e x for x ∈ R) < 2 2•(d+2)•2 2 √ d • 2 -2 -√ d •2 4 √ d (2 2 √ d ≥ d + 1 for d ≥ 0) = 2 2 2 √ d •(2d+4-2 √ d ) < 2 -2 2 √ d . (2d + 4 -2 √ d < -1 for d ≥ 42) (14) Similarly, P H∼P [∀b ∈ B ∀x ∈ X : x / ∈ path(h b ) ∧ h b (x) = 1] ≤ 2 - √ d•|B×X| ,(15)
so
P H∼P ∃B ⊆ {0, 1} d+1 ∃ X ⊆ T d : E(B, X, 0) ∧ |H| ≥ 2 2 √ d ∧ |X| ≥ 3 √ d ≤ |{0, 1} d+1 | 2 2 √ d |T d | 3 √ d • 2 - √ d•2 2 √ d •3 √ d (union bound, Eq. (15)) ≤ 2 d+1 2 2 √ d + 1 2 d+1 3 √ d + 1 • 2 -3d•2 2 √ d < 2 (d+1)• 2 2 √ d +1 • 2 (d+1)•(3 √ d+1) • 2 -3d•2 2 √ d ( n k < n k for k ≥ e) < 2 (d+1)• 2 2 √ d +3 √ d+2 • 2 -3d•2 2 √ d < 2 2d•2 2 √ d • 2 -3d•2 2 √ d (for d ≥ 4) < 2 -d2 √ d .(16)
Applying a union bound to Eqs. ( 14) and ( 16) gives
P H∼P [H satisfies Items 1 and 2] ≥ 1 -2 -2 2 √ d -2 -d2 √ d ≥ 1 -10 -100 .
In particular, there exists a collection H that satisfies Items 1 and 2. Furthermore, this collection has LD(H
) = d + 1 (namely, LD(H) ≥ d + 1 because it shatters T d ; and LD(H) ≤ d + 1 because |H| = 2 d+1 ).
this section cite: []

Section: D.3 Algorithm
In this section we describe Algorithms 5, 6a, and 6c, which together constitute the learning algorithm that achieves the O √ d mistake upper bound in the transductive setting, as in Theorem D.1. See Section 2.3 for a general overview of these algorithms.
this section cite: []

Section: D.3.1 How Experts Work
We start with some preliminary remarks about experts in Algorithms 5, 6a, and 6c.
this section cite: []

Section: Experts.
A tuple e = (S, u, H) defines an expert that can make predictions using the procedure EXPERT.PREDICT(e, •). The tuple e reflects two kinds of information:
1. Knowledge. Information that the expert knows with certainty. Specifically, this reflects the labels y 1 , y 2 , . . . sent by the adversary so far. All experts see the labels sent by the adversary, so this knowledge is the same for all experts. 2. Assumptions. At certain times, experts make assumptions about things that are not known for certain. Specifically, experts assume that certain nodes x are on-path (x ∈ path(h)) or off-path (x / ∈ path(h)) with respect to the correct labeling function h : T d → {0, 1}. Assumptions are simply guesses that may be wrong, and therefore when an expert needs to make such an assumption, it splits into two experts (as described below), with one expert assuming x ∈ path(h), and the other expert assuming x / ∈ path(h). This ensures that there always exists an expert for which all assumptions are correct.
In greater detail, the contents of the state tuple e = (S, u, H) represents the knowledge and assumptions of the expert as follows:
• u ∈ T d -This single node encodes everything the expert knows and assumes about which of the nodes labeled so far are on-path. Observe that if v 1 , v 2 , . . . , v k ∈ T d are nodes that are assumed to be on-path (and all these assumptions are consistent), then these k assumptions can be represented succinctly by assigning u = v i * where v i * is the deepest node among v 1 , v 2 , . . . , v k . Therefore, u simply holds the deepest node in the tree that is known or assumed to be on-path. At the start of the algorithm, this value is initialized to be u = λ, because the root is known to be on-path regardless of the target function.
• S ⊆ T d -the 'danger zone', as described in Section 2.3.4. This is a collection that contains all nodes in the prefix x ≤tmax = (x 1 , x 2 , . . . , x tmax ) of the sequence to be classified that have not been labeled yet and might be on-path for the true labeling function h given what the expert knows and assumes so far. However, S is not required to contain ancestors of nodes that are assumed to be on-path. Initially, S equals the prefix x ≤tmax . As information accumulates, nodes that cannot be on-path are removed from S. For instance, if x i ∈ T d is assigned label y i ∈ {0, 1} by the adversary, then any (1y i )-descendant of x i (including x i itself) may safely be removed from S.
• H ⊆ {0, 1} T d -the version space of the experts, i.e., the collection of all functions that could be the correct labeling function given everything that the expert knows and assumes. Initially, H contains all functions in H. As information accumulates, some functions are ruled out. Specifically, a function h can be removed from H for two reasons: (i) the adversary assigns a label y ̸ = h(x) to some node x ∈ T d ; (ii) the expert makes an assumption that some x ∈ T d is on-path for the correct labeling function but x / ∈ path(h), or vice versa, the expert assumes that x is off-path for the correct labeling function but x ∈ path(h).
this section cite: []

Section: Updates and splits.
An expert can be modified using the procedure EXPERT.EXTENDEDUPDATE(e, •, •). This procedure either returns a single modified tuple (S, u, H) (in the first two return statements in the procedure), in which case we think of the expert as being updated; or alternatively, the procedure returns two tuples e ∈ = (S ∈ , u ∈ , H ∈ ) and e / ∈ = (S / ∈ , u / ∈ , H / ∈ ) (in the third return statement), in which case we think of the expert as being split into two experts. The expert e ∈ corresponds to adding an assumption that the most recently presented node x t is on-path for the correct labeling function, and e / ∈ corresponds to adding the opposite assumption.
Ancestry. At the end of each iteration of the outer 'for' loop in Algorithm 5, for each expert e ∈ E t+1 there exists a unique ancestry sequence ancestry(e) = (e 1 , e 2 , . . . , e t+1 ) such that e 1 = ({x 1 , . . . , x tmax }, λ, H) is the initial single expert that was created before the start of the outer 'for' loop, e t+1 = e is the latest version of the expert, and for each i ∈ [t], the expert e i+1 was created by an execution of EXPERT.BASICUPDATE(e i , •, •) possibly followed by an execution of EXPERT.EXTENDEDUPDATE. 23
this section cite: []

Section: D.4 Analysis
In this section we prove our main result, Theorem D.1.
this section cite: []

Section: D.4.1 Assumption-Consistent Expert
Occasionally, when an expert is updated, it makes an assumption about whether the most-recently presented node x t is on-path or off-path with respect to the true labeling function h. In these updates, the expert is split into two: one expert assumes that x t ∈ path(h), and the other assumes x t / ∈ path(h). Clearly, by splitting into two in this manner, we preserve the invariant that the set of experts always contains a 'vindicated' expert e * such that all the assumptions made by e * are correct. This simple observation is made formal in the following definition and claim.
this section cite: []

Section: Definition D.3 (Assumption consistency).
For an expert e ∈ E t+1 with ancestry(e) = (e 1 , e 2 , . . . , e t+1 ), and an index i ∈ [t], we say that the i → (i + 1) update of e was assumption-consistent with a function h : T d → {0, 1} if one of the following conditions holds:
• e i+1 = EXPERT.BASICUPDATE(e i , x i , y i ); or 23 Note that in this paper, we use genealogical metaphors in two distinct contexts that should not be confused. First, as is customary, we use "child", "parent", "ancestor" and "descendant" to describe relations between nodes in the binary tree T d , which constitutes the domain of our hypothesis class. Separately from that, we use "ancestor" and "descendant" to describe relations between experts.
This overlap in terminology can partially be excused by the fact that the history of experts also forms a binary tree. Indeed, initially there is a single expert (the root of the tree), and experts can split into two, corresponding to a node having two children as in a binary tree. Seeing as experts cannot merge, the expert history corresponds precisely to a binary tree. (However, the domain T d is a perfect binary tree, whereas the binary tree corresponding to expert genealogy need not be balanced).
To reduce confusion, we use path(•) only for nodes in T d , and ancestry(•) only for experts, even though these operators are mathematically equivalent (however, path(•) is defined not only for nodes in T d but also for functions T d → {0, 1}).
• d, n ∈ N, λ is the empty string.
• H ⊆ {0, 1} T d is the class that exists by Lemma D.2.
• x 1 , x 2 , . . . , x n ∈ T d are points to be classified.
TRANSDUCTIVELEARNER(H, d, (x 1 , x 2 , . . . , x n )): t ← 0, t max ← 2 4 √ d e ← ({x 1 , . . . , x tmax }, λ, H) ▷ The initial expert. An expert is defined by a 3-tuple. w(e) ← 1 ▷ Assign the initial expert a weight of 1. E 1 ← {e} ▷ E t is the set of experts used for predicting ŷt . E 2 , . . . , E n , E n+1 ← ∅ for t ← 1, 2, . . . , n: ŷt ← 1 e∈Et w(S) • EXPERT.PREDICT(e, x t ) ≥ 1 2 ▷ A weighted majority, using Algorithm 6a. send prediction ŷt to adversary receive correct label y t ∈ {0, 1} from adversary for e ∈ E t : ▷ Update the experts. e ← EXPERT.BASICUPDATE(e, x t , y t ) ▷ Remove functions that disagree with the label y t from the version space. if EXPERT.PREDICT(e, x t ) = y t : E t+1 ← E t+1 ∪ {e} ▷ If expert e made a correct prediction, no further update is needed. else: U ← EXPERT.EXTENDEDUPDATE(e, x t , y t ) ▷ If e made a mistake, update e using Algorithm 6c. This might cause e to be split into two experts. for e ′ ∈ U : E t+1 ← E t+1 ∪ {e ′ } ▷ Add updated expert(s) to E t+1 . w(e ′ ) ← w(e)/(2 • |U |)
▷ When e makes a mistake, its weight is decreased by a factor of 2 and then split equally between its descendants.
Algorithm 5: A transductive online learning algorithm that makes at most O √ d mistakes. It is a variant of the multiplicative weights algorithm that employs splitting experts. Namely, we start with a single expert, and when an expert makes a mistake it may split into two experts. The behavior of the experts is defined in Algorithms 6a and 6c.
• e i+1 was the single expert returned when executing EXPERT.
EXTENDEDUPDATE(e ′ i , x i , y i ) for e ′ i = EXPERT.BASICUPDATE(e i , x i , y i ); or • Executing EXPERT.EXTENDEDUPDATE(e ′ i , x i , y i ) with e ′ i = EXPERT.BASICUPDATE(e i , x i , y i ) returned two experts (S ∈ , u ∈ , H ∈ ) and (S / ∈ , u / ∈ , H / ∈ ) (as in the third return statement), and furthermore,
e i+1 = (S ∈ , u ∈ , H ∈ ) x i ∈ path(h) (S / ∈ , u / ∈ , H / ∈ ) x i / ∈ path(h).(17)
Assumptions:
• d, x, e, S, u, H -as in Algorithm 6a.
• y -the correct label for x, as selected by the adversary.
EXPERT.EXTENDEDUPDATE(e, x, y):
(S, u, H) ← e ▷ Unpack the state that defines the expert.
if |H| ≤ 2 2 √ d :
▷ If the version space is small, we just simulate the Halving algorithm, so the update is complete. [Case III] return {(S, u, H)} for b ∈ {0, 1}: return {(S ′ , u, H)} else:
S b ← {x ′ ∈ S : x ≼ b x ′ } ▷ Set of suspected on-path nodes that are b- descendant of x. if |S (1-y) | > |S|/3: S ′ ← S \ S (1-y) ▷ At
S / ∈ ← S; u / ∈ ← u ▷ Split e in two.
First, construct e / ∈ to be an updated version of e after adding the assumption that x / ∈ path(h) for the correct labeling function h.
H / ∈ = {h ∈ H : x / ∈ path(h)} e / ∈ ← (S / ∈ , u / ∈ , H / ∈ ) S ∈ ← S 0 ∪ S 1 ▷ Next
, construct e ∈ to be an updated version of e adding the assumption x ∈ path(h). S ∈ contains only nodes that are descendants of x. u ∈ ← u ▷ u ∈ represents updating the prior assumption that u is on path by adding that x is also on path.
if u ∈ ≼ x: u ∈ ← x H ∈ = {h ∈ H : x ∈ path(h)} ▷ H ∈ is obtained by updating the version space to include only function where x is on path. e ∈ ← (S ∈ , u ∈ , H ∈ ) return {e / ∈ , e ∈ } ▷ [Cases V and VI] Algorithm 6c: A subroutine of Algorithm 5 that defines how an expert is updated (and possibly split into two) when it makes a mistake.
Proof. We prove by induction that, for all s ∈ [t + 1], E s contains a unique expert that is assumptionconsistent with h.
The base case s = 1 is clear, because E 1 contains only a single expert that was never modified. For the induction step, let e * s be the unique assumption-consistent expert in E s , and consider the s → (s + 1) update. Notice that by Definition D.3, • For all e ∈ E s \ {e * s }, every expert e ′ ∈ E s+1 such that e ′ was created from e by executing EXPERT.BASICUPDATE(e s , x s , y s ) possibly followed by an execution of EXPERT.EXTENDEDUPDATE is not assumption-consistent with h; and • Either EXPERT.BASICUPDATE(e * s , x s , y s ) ∈ E s+1 and EXPERT.EXTENDEDUPDATE(e * s , x s , y s ) is not executed (e * s is added to E s+1 with just a basic update), or precisely one of the experts that were created from e * s by executing EXPERT.EXTENDEDUPDATE and added to E s+1 is assumption-consistent with h. any function h : T d → {0, 1}. This establishes the base case for Item 1. Additionally, S * 1 = {x 1 , x 2 , . . . , x tmax }, satisfying the base case for Item 2. For the induction step, we assume that the claim holds for some integer t = i, and show that it holds for t = i + 1 as well. First, we establish Item 1. If e * i+1 = EXPERT.BASICUPDATE(e * i , x i , y i ), then the claim is immediate because u * i+1 = u * i ∈ path(h). Otherwise, by Definition D.3 and the first first two return statements in EXPERT.EXTENDEDUPDATE, either e * i+1 = (S * i+1 , u * i+1 , H * i+1 ) has u * i+1 = u * i ∈ path(h), in which case the claim is immediate, or else e * i+1 satisfies Eq. ( 17), namely,
e * i+1 = (S ∈ , u ∈ , H ∈ ) x i ∈ path(h) (S / ∈ , u / ∈ , H / ∈ ) x i / ∈ path(h).
As defined in EXPERT.EXTENDEDUPDATE, u ∈ is equal either to u * i or to
x i , so if x i ∈ path(h) then u * i+1 = u ∈ ∈ {u * i , x i } ⊆ path(h). On the other hand, if x i / ∈ path(h) then we get u * i+1 = u / ∈ = u * i ∈ path(h).
We see that in all cases, u * i+1 ∈ path(h) as desired. This concludes the proof of Item 1. For Item 2, again, if e * i+1 = EXPERT.BASICUPDATE(e * i , x i , y i ), then the claim is immediate because S * i+1 = S * i and u * i+1 = u * i . Otherwise, consider the various ways in which u * i+1 and S * i+1 can be assigned by EXPERT.EXTENDEDUPDATE. In the first return statement, u * i+1 = u * i and S * i+1 = S * i , and the claim is immediate. The second return statement assigns u * i+1 = u * i and S * i+1 = S * i \ S 1-yi , where S 1-yi is the set of (1y i )-descendants of x i (including x i itself). Notice that regardless of whether x i is on-path for the correct labeling function h or not, none of the (1y i )-descendants of x i (except possibly x i itself) can be on-path for h, because h assigns a label y i to x i . And seeing as Item 2 only requires that S * i+1 contain nodes from {x i+1 , x i+2 , . . . , x tmax }, it is also safe to remove x i . Therefore, removing S 1-yi preserves Item 2. For the third return statement, there are two possibilities. The first possibility is that u * i+1 = u / ∈ = u * i and S * i+1 = S / ∈ = S * i , in which case the claim is immediate. The second possibility assigns u * i+1 = u ∈ , and S * i+1 = S ∈ = S 0 ∪ S 1 , namely, S * i+1 is constructed by removing the nondescendants of x i from S * i . By Eq. ( 17), this happens when x i ∈ path(h), so all non-descendants of x i or either off-path for h, or they are ancestors of x i . Seeing as x i ∈ path(h) and u * i ∈ path(h), and u ∈ is the deeper node between these two, any node that is an ancestor of x i is also an ancestor of u * i+1 = u ∈ . Thus, all the nodes removed or either off-path for h, or they are ancestors of u * i+1 , satisfying Item 2. (Similarly, any node that is an ancestor of u * i is also an ancestor of u * i+1 , so we do not need to add any new nodes to S * i+1 that are not included in S * i .) We see that in all cases, Item 2 is preserved, as desired.
this section cite: []

Section: D.4.2 Transition to Halving
Claim D.6. Let d, n, t ∈ N, d ≥ 16, let H ⊆ {0, 1} T d , and let x 1 , . . . , x n ∈ T d . Consider an execution of TRANSDUCTIVELEARNER(H, (
x 1 , x 2 , . . . , x n ))
as in Algorithm 5. Let t > t max = 2 4 √ d and let e = (S, u, H) ∈ E t be an expert. Then
|H| ≤ 2 2 √ d . Proof of Claim D.6. Assume for contradiction that |H| > 2 2 √ d . Let H ′ ⊆ H be an arbitrary subset of size 2 2 √ d + 1. Let P = ∪ h∈H ′ path(h). Seeing as each root-to-leaf path contains d + 1 nodes, |P | ≤ |H ′ | • (d + 1) ≤ 2 2 √ d + 1 • (d + 1) ≤ d2 2 √ d+1 .(18)
Let y 1 , y 2 , . . . , y t be the labels provided by the adversary in the first t max iterations. The line in EXPERT.BASICUPDATE constructing H using HALVING.UPDATE(H, x, y) ensures that ∀h ∈ H ∀i ∈ [t max ] : h(x i ) = y i . (19) Consider two cases:
• Case I. tmax i=1 y i ≤ t max /2. Then the set X 0 = {x i : i ∈ [t max ] ∧ y i = 0} has cardinality |X 0 | ≥ t max /2. Let X ′ 0 = X 0 \ P . By Eq. (18), |X ′ 0 | ≥ t max 2 -d2 2 √ d+1 = 2 4 √ d -d2 2 √ d+1 .(20)
From the choice of X ′ 0 , the inclusion H ′ ⊆ H, and Eq. ( 19), ∀h ∈ H ′ ∀x ∈ X
′ 0 : x / ∈ path(h) ∧ h(x) = 0.(21)
Seeing as |H ′ | > 2 2 √ d , Eq. ( 21) and Item 1 from Lemma D.2 imply that
|X ′ 0 | ≤ 2 2 √ d .(22)
Combining Eqs. ( 20) and ( 22) yields
2 2 √ d ≥ |X ′ 0 | ≥ 2 4 √ d -d2 2 √ d+1 ≥ 2 4 √ d-1 (d ≥ 16),
which is a contradiction.
• Case II.
tmax i=1 y i > t max /2. A similar argument gives a contradiction by defining
X 1 = {x i : i ∈ [t max ] ∧ y i = 1}, and X ′ 1 = X 1 \ P. As before, |X ′ 1 | ≥ t max 2 -d2 2 √ d+1 ≥ 2 4 √ d -d2 2 √ d+1 . (23
) for all d ∈ N. However, |H ′ | > 2 2 √ d
and Item 2 imply that
|X ′ 1 | < 3 √ d,(24)
which is a contradiction.
this section cite: []

Section: D.4.3 Performance of Best Expert
Claim D.7 (Existence of expert with large weight). Let d, n ∈ N, d ≥ 16, let H ⊆ {0, 1} T d , and let x 1 , . . . , x n ∈ T d . Consider an execution of
TRANSDUCTIVELEARNER(H, (x 1 , x 2 , . . . , x n ))
as in Algorithm 5. Then, at the end of the execution, there exists e ∈ E n+1 such that
w(e) ≥ 2 -48 √ d .(25)
Note that the lower bound in Eq. ( 25) does not depend on n.
Proof. Fix a hypothesis h ∈ H such that h(x t ) = y t for all t ∈ [n] (such an h exists because the adversary must always select a realizable label).
By Claim D.4, there exists e * n+1 ∈ E n+1 that is assumption-consistent with h. Let ancestry e * n+1 = (e *
1 , e * 2 , . . . , e * n+1 ). We argue that this ancestry sequence makes few mistakes. Specifically, for each t ∈ [n], let ŷ * t = EXPERT.PREDICT(e * t , x t ). We claim that
m := n t=1 1(ŷ * t ̸ = y t ) ≤ 24 √ d.
Indeed, let B = {t ∈ [n] : ŷ * t ̸ = y t } be the set of m indices where a mistake was made. For each t ∈ B, let e * t = (S, u, H), and note that each t ∈ B has a corresponding execution of EXPERT.PREDICT(e * t , x t ), and an execution of e ′ t = EXPERT.BASICUPDATE(e * t , x t , y t ) followed by EXPERT.EXTENDEDUPDATE(e ′ t , x t , y t ) that produces e * t+1 (EXPERT.EXTENDEDUPDATE is executed because t ∈ B, i.e., a mistake was made). We partition the indices in B into six cases (six disjoint sets), and bound the number of indices that fall in each.
• Case I. The execution of EXPERT.PREDICT(e * t , x t ) exited via the first return statement in that procedure. This happens once |H| ≤ 2 2 √ d , and from that point on, the expert and all subsequent experts in the ancestry are exactly simulating the HALVING algorithm (Algorithm 7) in both predictions and updates. Hence, by Fact E.1, B contains at most m I = 2 √ d such indices.
• Case II. The execution of EXPERT.PREDICT(e * t , x t ) exited via the second return statement in that procedure. In particular x ≼ u, and the predicted label was ŷ * t = b ∈ {0, 1} such that x t ≼ b u. Because e * t is assumption-consistent with h, Item 1 in Claim D.5 implies that u ∈ path(h). Namely, we see that u is a b-descendant of x t and u ∈ path(h). It follows that ŷ * t = b = h(x t ) = y t . So no mistakes are made in Case II, and the number of indices t ∈ B that belong to Case II is simply m II = 0.
In the remaining cases, we assume that EXPERT.PREDICT(e * t , x t ) exited via the third return statement in that procedure, so the prediction was
ŷ * t = 1(|S 1 | > |S|/3),(26)
where S 1 = {x ′ ∈ S : x t ≼ 1 x ′ }. These cases are as follows.
• Case III.
The execution of EXPERT.EXTENDEDUPDATE(e ′ t , x t , y t ) exited via the first return statement in that procedure. Namely, after the update, the resulting expert e * t+1 has |H| ≤ 2 2 √ d . However, because we are not in Case I, at the beginning of the iteration expert e * t had |H| > 2 2 √ d . Seeing as the cardinality of H decreases monotonically throughout the ancestry e * 1 , . . . , e * n+1 , this type of mistake can happen at most m III = 1 times.
• Case IV. The execution of EXPERT.EXTENDEDUPDATE(e ′ t , x t , y t ) exited via the second return statement in that procedure. In this case, |S (1-yt) | > |S|/3, and e * t+1 = (S ′ , u, H) with S ′ = S \ S 1-yt . So |S ′ | < 2|S|/3. Namely, the update causes the cardinality of the set S to be multiplied by a factor of at most 2/3 and it strictly decreases. Seeing as the initial cardinality is t max , and cardinalities are integers, the number of times this can happen is at most
m IV = log(t max ) log(3/2) + 1 = 4 √ d log(3/2) + 1.(27)
In the remaining cases, we assume that the execution of EXPERT.EXTENDEDUPDATE(e * t , x t , y t ) exited via the third return statement in that procedure. This implies that
|S ŷ * t | ≤ |S|/3(28)
Combining this with Eq. ( 26), it follows ŷ * t = 0 and therefore y t = 1. The remaining cases are as follows.
• Case V. x t ∈ path(h). Let e * t = (S, u, H). Seeing as |H| > 2 2 √ d (because we are not in Case I), Claim D.6 (with the assumption d ≥ 16) implies that t ≤ t max . By Item 2 of Claim D.5, the facts x t ̸ ≼ u (we are not in Case II) and x t ∈ path(h) imply that x t ∈ S. In particular, S is not empty.
Because the t → (t + 1) update of e * t+1 was assumption-consistent with h, Eq. ( 17) implies that e * t+1 = (S ∈ , u ∈ , H ∈ ), with S ∈ = S 0 ∪ S 1 . Observe that • |S 0 | ≤ |S|/3 (plugging ŷ * t = 0 into Eq. ( 28)); and • |S 1 | ≤ |S|/3 (because otherwise, by Eq. ( 26), the prediction would have been ŷ * t = 1). Therefore,
|S ∈ | ≤ |S 0 | + |S 1 | ≤ 2|S|/3.(29)
As in Case IV, combining Eq. ( 29) and the fact that S is not empty imply an upper bound m V on the number of times Case V can happen, with the bound being the same number m V = m IV as in Eq. ( 27).
• Case VI. x t / ∈ path(h). So (x t , y t ) is a pair such that x t / ∈ path(h) and y t = 1. Assume for contradiction that this type of mistake can happen strictly more than m VI = 3 √ d times. Let t 1 , t 2 , . . . , t mVI be the indices of the first m VI iterations of the outer 'for' loop of TRANSDUCTIVELEARNER in which this type of mistake happened. Note that if at the end of iteration t mVI , we had expert e * tm VI +1 = (S tm VI +1 , u tm VI +1 , H tm VI +1 ) such that |H tm VI +1 | ≤ 2 2 √ d , then from that point onwards, the expert would be simulating the halving algorithm, and in particular, it would not make any further mistake of the type in Case VI (all subsequent mistakes would belong to Case I). Hence, by the assumption that strictly more than m VI mistakes were made, it follows that
|H tm VI +1 | > 2 2 √ d . Let H * = h ′ ∈ H : (∀t ∈ [m VI ] : h ′ (x tt ) = 1 ∧ x t / ∈ path(h ′ ))
.
Because e * tm VI +1 is assumption-consistent with h, and from the construction of H tm VI +1 using H ∈ and H / ∈ in EXPERT.EXTENDEDUPDATE, it follows that H tm VI +1 ⊆ H * . So there exist collections H * ⊆ H and X = {x tt : t ∈ [m VI ]} ⊆ T d such that
• |H * | ≥ |H tm VI +1 | > 2 2 √ d , • |X| = m VI = 3 √ d, • ∀h ′ ∈ H * ∀x ∈ X : h ′ (x) = 1. • ∀h ′ ∈ H * ∀x ∈ X : x / ∈ path(h ′ ).
This is a contradiction to the choice of H, specifically, to Item 2 in Lemma D.2.
Thus, combining the analyses of all cases, we see that the number of mistakes made by the ancestry e * n+1 is at most m ≤ m I + m II + m III + m IV + m V + m VI ≤ 2 √ d + 0 + 1 + 4 √ d log(3/2) + 1 + 4 √ d log(3/2) + 1 + 3 √ d ≤ 24 √ d. The weights satisfy w(e * t+1 ) = w(e * t ) ŷ * t = y t ≥ 1 4 • w(e * t ) ŷ * t ̸ = y t . This implies that w(e * n+1 ) ≥ w(e * 1 ) • n t=1 4 -1(ŷi̸ =yi) = w(e * 1 ) • 4 -m ≥ 4 -24 √ d = 2 -48 √ d , as desired.
D.4.4 Multiplicative Weights Mistake Bound Claim D.8 (Mistake bound for multiplicative weights). Let d, n ∈ N, let α > 0, let H ⊆ {0, 1} T d , and let x 1 , . . . , x n ∈ T d . Consider an execution of
TRANSDUCTIVELEARNER(H, (x 1 , x 2 , . . . , x n ))
as in Algorithm 5. Assume that at the end of the execution, there exists e * ∈ E n+1 such that w(e * ) ≥ 2 -α .
Then TRANSDUCTIVELEARNER makes at most α mistakes.
Proof of Claim D.8. For all i ∈ [n + 1], let w(E i ) = e∈Ei w(e). For each i ∈ [n], if ŷi ̸ = y i , then w(E i+1 ) ≤ w(E i )/2. Hence, if TRANSDUCTIVELEARNER makes m mistakes, then by induction
w(E n+1 ) ≤ w(E 1 ) • n t=1 2 -1(ŷi̸ =yi) = 2 -m • w(E 1 ). So 2 -α ≤ w(e * ) ≤ e∈En+1 w(e) = w(E n+1 ) ≤ 2 -m • w(E 1 ) = 2 -m .
We conclude that m ≤ α, as desired.
this section cite: []

Section: D.5 Proof
Proof of Theorem D.1. Fix an integer d ≥ 43. Let H ⊆ {0, 1}
T d-1 be the class constructed by invoking Lemma D.2 for the integer d -1 ≥ 42. We argue that this class satisfies the requirements of Theorem D.1. By construction, H is a class of Littlestone dimension precisely d. By Theorem A.7, this implies the equality in Item 2. We now show the upper bound in Item 1. We argue that TRANSDUCTIVELEARNER (Algorithm 5) satisfies this upper bound. By Claim D.7, at the end of the execution of TRANSDUCTIVELEARNER there exists an expert e ∈ E n+1 such that w(e) ≥ 2 -48 √ d . By Claim D.8, this implies that the number of mistakes made by TRANSDUCTIVELEARNER is at most 48 √ d, as desired.
this section cite: []

Section: E Halving
Fact E.1. Let X be a set, and let H ⊆ {0, 1} X be a hypothesis class. Then for all n ∈ N, all sequences x ∈ X n , and all realizable adversaries, HALVING (Algorithm 7) makes at most log(|H|) mistakes in the transductive online learning (Game 2). 25 Namely, sup n∈N sup A∈An M tr (H, n, HALVING, A) ≤ log(|H|). 25 With the suitable syntactic modification, it also makes at most log(|H|) mistakes in the standard online learning (Game 1).
this section cite: []

Section: 
Assumptions:
• d ∈ N, x ∈ T d .
• e = (S, u, H) is a tuple that defines an expert:
• S ⊆ T d -a collection of nodes that could be on-path for the true labeling function given what the expert knows and assumes.
• u ∈ T d -the deepest node known or assumed to be on-path by the expert.
• H ⊆ {0, 1} T d -the collection of all functions that could be the correct labeling function given what the expert knows and assumes.
EXPERT.PREDICT(e, x):
(S, u, H) ← e ▷ Unpack the state that defines the expert.
if |H| ≤ 2 2 √ d : return HALVING. PREDICT(H, x) ▷ Once H becomes small enough, simulate the Halving algorithm (Algorithm 7).
[Case I] if x ≼ u: return b ∈ {0, 1} such that x ≼ b u ▷ u is assumed to be on-path. If u is a bdecendant of x, then the correct label for x must be b. Algorithm 6a: A subroutine of Algorithm 5 that defines how an expert makes predictions.
[Case II] return 1(|{x ′ ∈ S : x ≼ 1 x ′ }| > |S|/3) ▷ Output
Assumptions:
• x, e, S, u, H -as in Algorithm 6a.
• y -the correct label for x, as selected by the adversary.
EXPERT.BASICUPDATE(e, x, y):
(S, u, H) ← e ▷ Unpack the state that defines the expert.
H ← HALVING.UPDATE(H, x, y) ▷ Update the version space, as in the Halving algorithm (Algorithm 7). return (S, u, H) Algorithm 6b: A subroutine of Algorithm 5 that defines how an expert is updated each time that a label is selected by the adversary.
We say that an expert e ∈ E t+1 is assumption-consistent with h if for all i ∈ [t], the i → (i + 1) update of e was assumption-consistent with h. Claim D.4 (Existence of assumption-consistent expert). Let d, n, t ∈ N, t ≤ n, let H ⊆ {0, 1} T d , let x 1 , . . . , x n ∈ T d , and let h : T d → {0, 1}. Consider an execution of TRANSDUCTIVELEARNER(H, d, (x 1 , x 2 , . . . , x n )) as in Algorithm 5. Then, at the end of the t-th iteration of the outer 'for' loop in TRANSDUCTIVE-LEARNER, there exists a unique expert e *
t+1 ∈ E t+1 that is assumption-consistent with h.
Assumptions:
• X a set, k ∈ N.
• H ⊆ {0, 1} X is a finite hypothesis class.
• x, x 1 , . . . , x k ∈ X , y ∈ {0, 1}.
HALVING(H, (x 1 , x 2 , . . . , x k )):
H 1 ← H for i ∈ [k]: ŷi ← HALVING.PREDICT(H, x i )
send prediction ŷi to adversary receive correct label y i ∈ {0, 1} from adversary
H i+1 ← HALVING.UPDATE(H i , x i , y i ) HALVING.PREDICT(H, x): return 1 1 |H| h∈H h(x) ≥ 1 2 HALVING.UPDATE(H, x, y): return h ∈ H : h(x) = y
Algorithm 7: This is the well-known halving algorithm. The experts in Algorithms 6a and 6c simulate this algorithm once their version space becomes small enough.
Seeing as the s → (s + 1) update executes EXPERT.BASICUPDATE and EXPERT.EXTENDEDUPDATE at most once for each e ∈ E s , it follows that E s+1 contains precisely one expert that is assumption-consistent with h.
An expert e = (S, u, H) that is assumption-consistent with the correct labeling function enjoys two simple properties. The first property is that the node u in the expert encodes correct information about which previously seen nodes are on-path for the correct labeling function.
The second property is that the set S contains all future nodes that are on-path for the correct labeling function and are also deeper in the tree than all nodes assumed to be on-path so far. These two properties are formalized in the following claim. Claim D.5 (Properties of assumption-consistent expert).
Let d, n, t ∈ N, t ≤ n + 1, let H ⊆ {0, 1} T d , let x 1 , . . . , x n ∈ T d . Consider an execution of TRANSDUCTIVELEARNER(H, d, (x 1 , x 2 , . . . , x n ))
as in Algorithm 5. Assume that the adversary selects labels y 1 , y 2 , . . . , y n ∈ {0, 1} that are consistent with some function h :
T d → {0, 1}. Let e * t = (S * t , u * t , H * t
) ∈ E t be the unique expert in E t that is assumption-consistent with h. 24 Then the following two properties hold:
1. u * t ∈ path(h). 2. {x ∈ {x t , x t+1 , . . . , x tmax } : x ∈ path(h) ∧ x ̸ ≼ u * t } ⊆ S * t .
Proof of Claim D.5. The proof proceeds by induction on t.
For the base case t = 1, E 1 contains a single expert e * 1 = (S * 1 , u * 1 , H * 1 ) where u * 1 = λ is the root of T d . Indeed, λ ∈ path(h) for NeurIPS Paper Checklist 1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The main claims made in the abstract and introduction accurately reflect the paper's contributions and scope. Guidelines: • The answer NA means that the abstract and introduction do not include the claims made in the paper. • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper. 2. Limitations Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [NA] Justification: Purely rigorous mathematical results. We explain precisely what our proofs imply (and therefore also what they do not imply).
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: For each theoretical result, the paper provides the full set of assumptions and a complete (and correct) proof.
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
Answer: [NA] Justification: The paper has no experiments.
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
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [NA] Justification: The paper does not include experiments requiring code.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [NA] Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [NA] Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [NA] Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: The research conducted in the paper conforms, in every respect, with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10. Broader impacts Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [No] Justification: The work is purely theoretical with no immediate direct societal impacts forseeable.
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
Answer: [NA] Justification: The paper poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [NA]
Justification: The paper does not use existing assets.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets ( if applicable). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMS as any important, original, or non-standard components. Guidelines:
• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
• Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: A discriminative model for semi-supervised learning Year: (2010)
Ref_id:b1 Title: Online learning versus offline learning Year: (1995)
Ref_id:b2 Title: Online learning versus offline learning Year: (1997)
Ref_id:b3 Title: Learning low-density separators Year: (2008)
Ref_id:b4 Title: Learnability with respect to fixed distributions Year: (1991)
Ref_id:b5 Title: Combining labeled and unlabeled data with co-training Year: (1998)
Ref_id:b6 Title: A theory of universal learning Year: (2021)
Ref_id:b7 Title: Efficient transductive online learning via randomized rounding Year: (2013)
Ref_id:b8 Title: Transductive inference for estimating values of functions Year: (1999-12-04)
Ref_id:b9 Title: Semi-Supervised Learning Year: (2006)
Ref_id:b10 Title: Unlabeled data does provably help Year: (2013-03-02)
Ref_id:b11 Title: Learning by transduction Year: (1998)
Ref_id:b12 Title: When can unlabeled data improve the learning rate? Year: (2019-06-28)
Ref_id:b13 Title: A trichotomy for transductive online learning Year: (2023)
Ref_id:b14 Title: Multiclass transductive online learning Year: (2024)
Ref_id:b15 Title: Online learning: A comprehensive survey Year: (2021)
Ref_id:b16 Title: Transductive inference for text classification using support vector machines Year: (1999)
Ref_id:b17 Title: From batch to transductive online learning Year: (2005)
Ref_id:b18 Title: Learning quickly when irrelevant attributes abound: A new linear-threshold algorithm Year: (1987)
Ref_id:b19 Title: Understanding Machine Learning: From Theory to Algorithms Year: (2014)
Ref_id:b20 Title: Estimation of Dependencies Based on Empirical Data. Nauka, Moscow Year: (1979)
Ref_id:b21 Title: Estimation of Dependences Based on Empirical Data. Springer, 2nd edition Year: (2006)
Ref_id:b22 Title: Semi-supervised learning literature survey Year: (2005)
