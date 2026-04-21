Title: Clustering via Hedonic Games: New Concepts and Algorithms
Abstract: We study fundamental connections between coalition formation games and clustering, illustrating the cross-disciplinary relevance of these concepts. We focus on graphical hedonic games where agents' preferences are compactly represented by a friendship graph and an enmity graph. In the context of clustering, friendship relations naturally align with data point similarities, whereas enmity corresponds to dissimilarities. We consider two stability notions based on single-agent deviations: local popularity and local stability. Exploring these concepts from an algorithmic viewpoint, we design efficient mechanisms for finding locally stable or locally popular partitions. Besides gaining theoretical insight into the computational complexity of these problems, we perform simulations that demonstrate how our algorithms can be successfully applied in clustering and community detection.

Section: Introduction
Coalition formation and clustering are two research areas that capture different aspects of grouping a set of entities in a meaningful or optimal way. While coalition formation studies autonomous agents who have preferences over the various ways in which they can be partitioned into groups, clustering aims to unearth hidden structure and similarities within a set of datapoints. Despite the shared theme of group formation, there is no substantial interplay between these two research fields.
This study aims to investigate how different solution concepts emerging from coalition formation-in particular, from hedonic games-can be used in data clustering. Traditional clustering metrics measure the quality of the boundaries between the clusters or the average similarities within clusters. We propose a criterion that is based on a form of stability: we aim for a clustering where no deviation by a single agent (or data point) should lead to a more preferred outcome. We aim to investigate how such stability notions can be harnessed in clustering algorithms.
Similarities and dissimilarities, the baseline concepts used by all clustering algorithms, can be associated with relations of compatibility and of conflict between two agents. Leveraging this idea, we focus on a model in coalition formation due to Dimitrov et al. [16] where agents' preferences over different coalitions are determined by their friendships and their enmities, represented as two graphs. More precisely, we adopt the model of hedonic friends-enemies-neutrals (FEN) games [33,37,27] where agents classify all other agents as a friend, an enemy, or a neutral, and each agent evaluates a given partition based on the number of friends and the number of enemies in their coalition. Unfortunately, finding "optimal" or, in some sense, stable partitions often turns out to be a computationally intractable task. Since computational efficiency is often crucial in clustering applications, we focus our attention on local stability concepts requiring only that re-assigning a single agent to a different coalition does not yield a partition deemed better for the majority of agents. Such single-agent deviations were studied in FEN games by Kerkmann and Rothe [28] and, more recently, in friends-enemies (FE) games-a model excluding neutrality-by Brandt et al. [10] who showed that in a model where two partitions are compared by taking the majority vote of the agents, such deviations converge into a locally stable partition in a polynomial number of rounds. We extend these results to FEN games and also to a slightly more stringent stability concept, local popularity. As opposed to the FE games studied by Brandt et al. [10], the option of neutrality introduces a flexibility that allows focusing on relations between points that are either very similar, or very different. Furthermore, neutrality yields smaller degrees in the friendship and enmity graphs which, as we show, has a huge benefit in terms of the convergence speed. Depending on the application under focus, different emphasis might be given to grouping together similar data points versus separating dissimilar data points. In the context of a FEN game, agents may put a significantly larger weight on the objective of being grouped together with friends than being separated from enemies, or vice versa; a third, more balanced approach is when agents treat these two objectives with the same importance. These options give rise to three preference domains-friend-appreciating, enemy-averse, and balanced-whose study may facilitate the application of our results to different clustering domains.
By identifying "similarity" with "friendship" and "difference" with "enmity," we demonstrate via simulations how our algorithms for finding locally stable or locally popular partitions can be used in the context of data clustering. The stability property enforced by our algorithm is particularly useful in clustering tasks where agents have meaningful local interactions or preferences. This includes standard benchmarks such as citation networks (e.g., Cora [36]), email communication graphs (e.g., Enron [30]), and social networks, where misclassifying a node can distort community structure. By ensuring that no agent prefers to switch to a different cluster based on local relationships, our method yields clusters that are robust against local deviations. This distinguishes our approach from traditional methods like k-means or DBSCAN, which may prioritize global fit over local consistency. As our simulations attest, this interdisciplinary approach can be surprisingly successful when compared to the most prevalent clustering algorithms. Thus we hope our work can spark a stronger cooperation between these fields to explore the applicability of coalition formation games in clustering.
this section cite: ['b15', 'b32', 'b36', 'b26', 'b27', 'b9', 'b9', 'b35', 'b29']

Section: Related work.
Although both hedonic games and clustering are widely studied, broad subjects, the only work connecting these two areas we are aware of is by Feldman et al. [20] who study hedonic clustering games in a noncooperative framework where clustering is obtained via decisions made by independent agents. They propose two types of hedonic games for which they characterize Nash equilibria and analyze the price of anarchy and the price of stability. Although some ideas appear both in our model and theirs-e.g., the utilities of agents depending on their similarities with agents within their own cluster and on dissimilarities with agents in other clusters-the models proposed by Feldman et al. significantly differ from ours. Most importantly, while our work is rooted in cooperative game theory, Feldman et al. take a noncooperative approach. In addition, although they establish the existence of Nash equilibria for certain specific domains, their results do not seem to lead to efficient, scalable algorithms for general clustering.
Regarding the broader literature related to our study, we only point out the most important precursors of our work and a few key references for more background on the relevant research areas. Coalition formation is a central topic in cooperative game theory. Starting with the seminal work of Drèeze and Greenberg [17], researchers' focus shifted to so-called hedonic games where agents' preferences depend only on the coalition they are contained in; see also [5,8,13] and the book chapters by Aziz and Savani [4] and Bullinger et al. [11,Section 3.6]. FE games were introduced by Dimitrov et al. [16] as a subclass of hedonic games with a concise description (namely, the agents' friendship relations). FEN games, the extension of their model that allows for neutrality, have also been studied [14,27,28,33,37,40]. Single-agent deviations and their induced dynamics were investigated in FE games by Brandt et al. [10] who, in various settings, showed that such deviations converge to a partition that fulfills certain stability concepts. Among the stability concept they considered is what we call local stability; our work extends their results on this notion to FEN games. Our stability concepts stem from the idea of popularity as introduced by Gärdenfors [22]. Popularity has been widely studied in matching and allocation problems [1,6,15,25,26] and has also been considered in the context of hedonic games by, e.g., Aziz et al. [3], Kerkmann et al. [27], Brandt and Bullinger [9], Kerkmann and Rothe [29], and Bullinger and Gilboa [12].
For a recent survey on clustering-a fundamental problem in unsupervised learning aimed at discovering structure in data by grouping similar elements-see [19]. Classical clustering methods such as k-means, introduced by MacQueen [34], optimize intra-cluster similarity under geometric assumptions, while DBSCAN by Ester et al. [18] identifies dense regions in data and is robust to noise. The most relevant subfield of clustering for our purposes is community detection, which seeks to partition the nodes of a graph into densely connected groups with sparse interconnections. Here, the Louvain algorithm of Blondel et al. [7] and its improved variant Leiden by Traag et al. [41] are state-of-the-art methods that optimize modularity to reveal community structure in networks. Our main metric to compare our algorithms with k-means, DBSCAN, Louvain, and Leiden is the Rand distance [39], which-assuming some true labeling for the data points-computes the fraction of pairs of points that are correctly clustered together or separately.
Our contribution. We study two stability concepts in hedonic FEN games: local popularity and local stability. The former-introduced in this paper-assures that no single agent can deviate such that more agents improve than get worse, while the latter requires this only for Nash deviations (where the deviating agent improves). Our main result is a simple algorithm that always finds a locally popular partition in polynomial time if friendship and enmity relationships are symmetric in all three settings (friend-appreciative, enemy-averse, and balanced). For asymmetric relationships, we show that a locally stable partition can still be found efficiently in the balanced case, but all other settings yield NP-hardness. See Table 1 for a summary of our results.
We also conduct an experimental study comparing our algorithms to common clustering and community detection algorithms, in particular, to k-means, DBSCAN, Louvain, and Leiden. Our simulations provided sufficient data to show that our algorithms are well suited for clustering and community detection tasks, often outperforming existing solutions.
this section cite: ['b19', 'b16', 'b4', 'b7', 'b12', 'b3', 'b10', 'b15', 'b13', 'b26', 'b27', 'b32', 'b36', 'b39', 'b9', 'b21', 'b0', 'b5', 'b14', 'b24', 'b25', 'b2', 'b26', 'b8', 'b28', 'b11', 'b18', 'b33', 'b17', 'b6', 'b40', 'b38']

Section: Preliminaries
Let N denote the set of agents. A subset C ⊆ N is called a coalition; in particular, N is the grand coalition. In this paper, partition refers to a family of pairwise disjoint nonempty subsets of N whose union is N . Given a partition π, the coalition in π that contains i ∈ N is denoted by π(i).
A hedonic game is a pair (N, ⪰), where ⪰ = (⪰ i ) i∈N is a preference profile that contains for each agent i a weak preference list ⪰ i over the possible coalitions containing i. That is, for any two coalitions C, C ′ with i ∈ C ∩ C ′ , either C ′ ≻ i C (meaning that C ′ is preferred to C by i), or C ≻ i C ′ (meaning that C is preferred to C ′ by i), or C ′ ∼ i C (meaning that i is indifferent between the two coalitions). In a hedonic game, the preference lists of the agents only depend on the coalitions they are in. Hence, ⪰ i also induces a weak ranking of agent i over the possible partitions.
Given an agent i ∈ N , a partition π, and a coalition C ∈ π ∪ {∅}, the deviation where i leaves its original coalition π(i) and joins C (or forms a new coalition {i} in the case when C = ∅) is referred to as the switch i → C. We use π i→C to denote the partition resulting from π by the switch i → C, that is, π i→C = (π \ {π(i), C}) ∪ {π(i) \ {i}, C ∪ {i}}. We say that i has an incentive to join C if C ∪ {i} ≻ i π(i); in this case, i → C is a Nash deviation. More generally, extending this notion to partitions, we say that an agent j votes for (or, j votes against) a switch i → C if j prefers π i→C to π (or, j prefers π to π i→C , respectively). Further, we define vote j (π, π ′ ) =    +1 if π ′ (j) ≻ j π(j), -1 if π(j) ≻ j π ′ (j), 0 if π(j) ∼ j π ′ (j)
for partitions π and π ′ over N . Let Λ(π, π ′ ) = j∈N vote j (π, π ′ ) be the sum of the agents' votes for π ′ against π. We will further write Λ -i (π, π ′ ) = j∈N \{i} vote j (π, π ′ ) for the sum of votes for π ′ against π among all agents excluding agent i.
this section cite: []

Section: Definition 1.
Given a partition π, an agent i, and a coalition C ∈ π ∪ {∅}, we say that the switch i → C locally dominates π if Λ(π, π i→C ) > 0; and the switch i → C locally blocks π if it is a Nash deviation that locally dominates π. A partition π is popular if for any partition π ′ , we have that Λ(π, π ′ ) ≤ 0. Moreover, π is locally popular if there is no switch that locally dominates π, and π is locally stable if there is no switch that locally blocks π.
Note that a single deviation by one agent can also change the situation-and thus the voting behavior on these two partitions-of other agents, giving rise to the above notions of local popularity and local stability. Note further that local popularity implies local stability.
Preferences in FEN games. Dimitrov et al. [16] introduced a preference domain for hedonic games where agents' preferences over the coalitions containing them depend only on the number of friends and enemies they have in their coalition. Thus, in a FEN game over agent set N , each agent i classifies all remaining agents into one of three disjoint categories: those that i considers either friends, enemies, or neutrals; we denote these agent sets by F (i), E(i), and N (i), respectively. The friendship and enmity relations can be represented by two directed graphs, the friendship digraph G F = (N, A F ) and the enmity digraph G E = (N, A E ), where an arc (i, j) ∈ A F means that i considers j a friend, whereas an arc (i, j) ∈ A E means that i considers j an enemy. We denote by ∆ the maximum number or arcs incident to any agent in G F or in G E . A FEN game is symmetric if friendship and enmity relations are both symmetric; it is asymmetric otherwise.
FEN games are subclasses of so-called additively separable hedonic games [5] where each agent i has a valuation function v i over N \ {i} that is defined as
v i (j) =
α f if i considers j a friend, i.e., j ∈ F (i), α e if i considers j an enemy, i.e., j ∈ E(i), 0 if i considers j a neutral, i.e., j ∈ N (i) for some integers α f > 0 and α e < 0, and agent i's utility in a given coalition C containing i is defined as u i (C) = j∈C\{i} v i (j); we refer to the pair (α f , α e ) as the type of the FEN game. Then the preferences of agent i are based on i's utilities: for two coalitions C and C ′ both containing i, C ⪰ i C ′ if and only if u i (C) ≥ u i (C ′ ). For a coalition C ⊆ N with i ∈ C, we let f i C = |F (i) ∩ C| and e i C = |E(i) ∩ C| denote the number of i's friends and enemies within C, respectively; then
u i (C) = f i C • α f + e i C • α e . The utility of agent i in a partition π is u i (π) = u i (π(i)).
Friend-appreciating, enemy-averse, and balanced preferences. Following Dimitrov et al. [16], we introduce the following preference domains for FEN games. A FEN game with appreciation of friends (or a FEN-AF game, for short) is a FEN game over agent set N whose type is (|N |, -1). That is, C ⪰ i C ′ holds for two coalitions C, C ′ with i ∈ C ∩ C ′ if and only if either f i C > f i C ′ , or f i C = f i C ′ and e i C ≤ e i C ′ . By contrast, a FEN game with aversion to enemies (or a FEN-AE game, for short) is a FEN game over agent set N whose type is (1, -|N |). That is, C ⪰ i C ′ holds for two coalitions C, C ′ with i ∈ C ∩ C ′ if and only if either e i C < e i C ′ , or e i C = e i C ′ and f i C ≥ f i C ′ . Finally, a FEN game with balanced preferences (or a FEN-B game, for short) is a FEN game with type (1, -1).
That is, C ⪰ i C ′ holds fo two coalitions C, C ′ with i ∈ C ∩ C ′ if and only if f i C -e i C ≥ f i C ′ -e i C ′ . We will call u B i (C) = f i C -e i C the balanced utility of i in C. Problem definitions. Let us define the decision problems related to the existence of locally popular and locally stable partitions.
Given a FEN game I of type (α f , α e ) defined over a set N of agents with a friendship digraph G F and an enmity digraph G E , the LOCAL-POPULARITY-EXISTENCE problem asks if there exists a locally popular partition for I; analogously, the LOCAL-STABILITY-EXISTENCE problem-given the same input-asks if there exists a locally stable partition for I.
this section cite: ['b15', 'b4', 'b15']

Section: Local heuristics
We present two local heuristics, one to find a locally popular partition, the other to find a locally stable partition. Both heuristics start from an arbitrary partition and iteratively modify it using only single-agent deviations; the conditions that determine whether or not to perform a given switch is a local condition that can be checked efficiently.
this section cite: []

Section: LocPop:
1. Start from an arbitrary partition π.
this section cite: []

Section: 2.
While there exists a coalition C ∈ π ∪ {∅} and an agent i / ∈ C such that Λ(π, π i→C ) > 0, replace π with π i→C and continue.
this section cite: []

Section: Table 1: Summary of our results on the computational complexity of LOCAL-POPULARITY-EXISTENCE and LOCAL-STABILITY-EXISTENCE.
We remark that a locally stable or locally popular partition always exists in the settings where the corresponding problem is proven to be in P. By contrast, a locally stable or locally popular partition may not exist in those settings where the corresponding problem is proven to be NP-complete (abbreviated as "NP-c").
LOCAL-STABILITY-EXISTENCE LOCAL-POPULARITY-EXISTENCE symmetric asymmetric symmetric asymmetric FEN-B game P (Thm. 2) P (Thm. 4) P (Thm. 2) NP-c (Thm. 9) FEN-AF game P (Thm. 2) NP-c (Thm. 7) P (Thm. 2) NP-c (Thm. 7) FEN-AE game P (Thm. 2) NP-c (Thm. 13) P (Thm. 2) NP-c (Thm. 13) LocStab: 1. Start from an arbitrary partition π.
2. While there exists a coalition C ∈ π ∪ {∅} and an agent i / ∈ C such that Λ(π, π i→C ) > 0 and C ∪ {i} ≻ i π(i), replace π with π i→C and continue.
LocPop and LocStab can both be modified such that the number of coalitions within the partition remains fixed: it suffices to consider switches of the form i → C where C ∈ π and {i} / ∈ π in Step 2.
In the rest of this section, we show that the above heuristics converge in certain FEN games, that is, they arrive at a locally popular or locally stable partition in a finite number of steps. We remark that LocPop and LocStab cannot guarantee to achieve popularity; in fact, we show in Appendix A that deciding whether there exists a popular partition-or even just verifying popularity-is coNP-hard.
Our convergence results both for symmetric and asymmetric FEN games rely on a potential function that we define as the total balanced utility of agents:
f (π) = i∈N u B i (π(i)) = i∈N f i π(i) -e i π(i) .
this section cite: []

Section: Symmetric FEN games
Lemma 1. For a FEN game I over agent set N , let π be a partition, i ∈ N an agent, and C ∈ π ∪{∅} a coalition with Λ(π, π i→C ) > 0. Then (a
) j∈N \{i} u B j (π i→C ) -u B j (π) = Λ -i (π i , π i→C ). Moreover, if I is a symmetric FEN game, then (b) u B i (C) -u B i (π(i)) = Λ -i (π i , π i→C ) and (c) f (π i→C ) = f (π i ) + 2Λ -i (π i , π i→C ).
Proof. First, consider the agents in N \ {i}. Each such agent j may gain or lose a friend or an enemy by the switch i → C, and thus its balanced utility increases exactly by the value of vote j (π, π i→C ) ∈ {-1, 0, 1} as a result of the switch i → C. Summing up the increase in the balanced utility of all agents in N \ {i}, we therefore obtain Λ -i (π, π i→C ), proving statement (a) of the lemma.
Assume now that I is symmetric. Consider the contribution of agent i to f (π i→C ) -f (π). The change in the balanced utility of i as a result of the switch i → C is exactly u B i (C) -u B i (π(i)) = Λ -i (π, π i→C ) as friendships are symmetric, so each agent who gains (or loses) a friend or an enemy due to i → C is in turn a friend or an enemy gained (or lost) by i; this yields statement (b).
this section cite: []

Section: Statement (c) then follows by summing up (a) and (b).
Theorem 2. In a symmetric FEN-AF, FEN-AE, or FEN-B game, irrespective of the initial partition, heuristic LocPop always converges in O(n∆ 2 ) steps (in O(n∆) steps for FEN-B games) to a locally popular (and thus also locally stable) partition, and heuristic LocStab always converges in O(n∆) steps to a locally stable partition. Proof. Consider a step in heuristic LocPop or LocStab where a partition π is replaced by π i→C for some agent i ∈ N and coalition C ∈ π ∪ {∅}. Then Λ(π, π i→C ) > 0 implies Λ -i (π, π i→C ) ≥ 0. By statement (b) of Lemma 1, this immediately yields that f (π i→C ) ≥ f (π), meaning that f is nondecreasing during the execution of the heuristic. As the total balanced utility is within the range [-n∆, n∆], there can be at most 2n∆ steps when f (π) strictly increases.
Let us now consider the case when f (π i→C ) = f (π), i.e., the potential function does not increase; our aim is to bound the number of such steps during the execution of the heuristic. By statement (b) of Lemma 1, by f (π i→C ) = f (π) we must have Λ -i (π, π i→C ) = 0. By statement (a) of Lemma 1,
Λ -i (π, π i→C ) = f i C -e i C -f i π(i) + e i π(i) = 0.(1)
We introduce g F (π) = |{(i, j) ∈ A F : π(i) = π(j)}| and g E (π) = -|{(i, j) ∈ A E : π(i) = π(j)}|.
We distinguish between the following three cases depending on the type of the FEN game.
Case 1: Balanced preferences. By Lemma 1, the switch i → C increases the balanced utility of i by exactly Λ -i (π, π i→C ) = 0. Since agent i has balanced preferences, this implies vote i (π, π i→C ) = 0. However, then Λ(π, π i→C ) = 0 follows, contradicting our assumption that the condition Λ(π, π i→C ) > 0 holds. This proves f (π i→C ) > f (π) for balanced preferences.
Case 2: Friend-appreciating preferences. Since we assumed friendships to be symmetric, g F (π i→C )-g F (π) = 2f i C -2f i π(i) . We claim that g F (π i→C ) > g F (π). Indeed, as the switch i → C was performed by the heuristic based on the condition Λ(π, π i→C ) > 0, it follows that agent i must prefer π i→C to π; hence, either f i C > f i π(i) holds as promised, or f i C = f i π(i) and e i C < e i π(i) ; however, (1) excludes the latter.
Case 3: Enemy-averse preferences. Since we assumed enmities to be symmetric, we have
g E (π i→C ) -g E (π) = 2e i π(i) -2e i C . We claim that g E (π i→C ) > g E (π).
As in Case 2, we have Λ(π, π i→C ) > 0, which implies that agent i must prefer π i→C to π; hence, either e i C < e i π(i)
holds as promised, or e i C = e i π(i) and f i C > f i π(i) ; however, (1) again excludes the latter. Total number of steps. We have shown that the number of steps when the potential function f strictly increases is at most 2n∆. In each such step, g F and g E might each decrease by at most ∆, yielding an upper bound of 2n∆ 2 on the total decrease during the algorithm. Since g F takes values within [0, n∆] and g E between [-n∆, 0], the number of steps when f (π) does not increase is at most 2n∆ 2 + n∆. This gives a bound of O(n∆ 2 ) on the number of steps taken by the heuristics.
It remains to show that heuristic LocStab (and LocPop for FEN-B games) converges in O(n∆) steps. In FEN-B games, f always strictly increases, as we showed, hence we need O(n∆) steps. Otherwise, in FEN-AE and FEN-AF games, a switch
i → C is only performed in LocStab if f i C ≥ f i π(i) (or if e i
C ≤ e i π(i) ); hence g F (or g E , respectively) never decreases. Thus the number of steps when f does not increase is at most n∆ in FEN-AF and FEN-AE games, proving the claim for LocStab.
By applying an efficient way to decide if a switch is locally dominating and by using a carefully maintained data structure for quickly finding such switches, our heuristics can be implemented in near-linear time if the maximum degree ∆ of the friendship and enmity digraphs is constant; see Appendix B. As efficiency is often paramount in clustering, we believe that this result significantly widens the possibilities for applying our heuristics in practical clustering instances. Theorem 3 is proven in Appendix B, while all further results marked with (⋆) are proven in Appendix D.
this section cite: []

Section: Theorem 3 (⋆).
LocPop can be implemented to run in O(n∆ 3 log n) time (or O(n∆ 2 log n) for FEN-B games) and LocStab can be implemented to run in O(n∆ 2 log n) time.
this section cite: []

Section: Asymmetric FEN games
The following example shows that heuristics LocPop and LocStab may not stop in an asymmetric FEN game with friend-appreciating or enemy-averse preferences.
this section cite: []

Section: Example 1.
In an asymmetric FEN-AF or FEN-AE game, heuristics LocPop and LocStab can run into a cycle and never stop. Let there be seven agents: 0, 1, . . . , 6. Each agent i considers (i + 1) mod 7 a friend but (i + 2) mod 7 and (i + 3) mod 7 enemies; see Figure 1a for an illustration. (Here and anywhere else, all relations that are not explicitly mentioned are tacitly assumed to be neutral.)
In the AF domain, let the initial partition be {{0, 1, 2, 3}, {4, 5, 6}}. Then agent 3 has an incentive to join {4, 5, 6} and this switch takes place, as agents 0, 1, and 3 vote for it and only agent 2 votes against it. By symmetry, this creates a loop: agent 6 switches next, then agent 2, then agent 5, etc.
In the AE domain, let the initial partition be {{0, 1}, {2, 3}, {4, 5, 6}}. Then agent 4 has an incentive to join {2, 3} and this switch takes place, as agents 4 and 3 vote for it and only agent 2 votes against it. Again, this creates a loop: agent 2 switches next, then agent 0, then agent 5, etc.
Example 2. In an asymmetric FEN-B game, heuristic LocPop can run into a cycle and never stop. Let the agent set be {0, 1, 2}, where each agent i considers (i -1) mod 3 a friend and (i + 1) mod 3 an enemy. Let the initial partition consist of two coalitions: {0, 1} and {2}. Then both 0 and 2 vote  for the switch of 1 → {2}, which hence takes place. Next, the switch 2 → {0} takes place, followed by the switch 0 → {1}, and the heuristic arrives back at the initial partition.
Contrasting Examples 1 and 2, heuristic LocStab always stops if agents' preferences are balanced. Theorem 4 (⋆). In a possibly asymmetric FEN-B game, the total balanced utility of all agents strictly increases in each step of heuristic LocStab; consequently, heuristic LocStab always converges in O(n∆) steps to a locally stable partition, irrespective of the initial partition.
4 Hardness results for local popularity and local stability Friend-appreciating FEN games. Let us introduce the gadget which lies at the heart of our hardness results on FEN-AF games; see Figure 1b for an illustration. Definition 2. Let k ≥ 5 be an integer with (2k + 1) mod 3 ̸ = 0. A set {0, 1, . . . , 2k} of agents forms an AF-gadget of size 2k + 1 if each agent i considers (i + 1) mod (2k + 1) a friend (so that the friendship graph is a directed cycle) and considers (i + 2), (i + 3), . . . , (i -4) mod (2k + 1) enemies.
this section cite: []

Section: Theorem 5 (⋆). The asymmetric FEN-AF game consisting solely of an AF-gadget admits no locally popular partition.
Building on AF-gadgets, a more involved construction shows that even a locally stable solution may fail to exist. In fact, Theorem 7 shows that it is NP-hard to decide whether a locally popular or a locally stable partition exists; our reductions are from 5-SAT and make ample use of AF-gadgets.
this section cite: []

Section: Theorem 6 (⋆).
There exists an asymmetric FEN-AF game that admits no locally stable partition.
this section cite: []

Section: Theorem 7 (⋆). LOCAL-POPULARITY-EXISTENCE and LOCAL-STABILITY-EXISTENCE are NPcomplete for FEN-AF games.
Balanced FEN games. Turning our attention to balanced FEN games, recall that given a FEN-B game I, heuristic LocStab always finds a locally stable partition for I in polynomial time, as stated in Theorem 4. By contrast, we introduce a gadget that is similar to the AF-gadget from Definition 2 but is tailored for showing that FEN-B games may not admit a locally popular partition. Relying on Bgadgets, by a reduction from the NP-complete problem 3-COLORING we can show the NP-hardness of the related decision problem, as stated in Theorem 9. Definition 3. A set {0, 1, . . . , 6} of agents forms a B-gadget if each agent i considers (i + 1) mod 7 and (i + 2) mod 7 a friend and considers (i + 3), (i + 4), (i + 5) mod 7 enemies.
this section cite: []

Section: Theorem 8 (⋆).
The asymmetric FEN-B game consisting solely of a B-gadget admits no locally popular partition.
this section cite: []

Section: Theorem 9 (⋆). LOCAL-POPULARITY-EXISTENCE is NP-complete for FEN-B games.
As we have shown in Theorem 4, heuristic LocStab increases the agents' total balanced utility-the potential function f -in each step when running on a FEN-B game. Hence, a partition π maximizing f (π) is locally stable but, as Theorem 8 shows, not necessarily locally popular. Interestingly, not even a popular partition is guaranteed to reach the maximum total balanced utility, even in symmetric FEN-B games; see Proposition 10. Moreover, deciding whether there exists a partition that achieves a given total balanced utility is NP-hard, as shown by Theorem 11.
this section cite: []

Section: Proposition 10 (⋆).
There exists a symmetric FEN-B game that admits a partition that is popular but does not maximize the agents' total balanced utility.
this section cite: []

Section: Theorem 11 (⋆).
Given a symmetric FEN-B game I and an integer t, the problem of deciding whether I admits a partition π whose total balanced utility is at least t is NP-complete.
this section cite: []

Section: Enemy-averse FEN games.
Similarly to the gadgets from Definitions 2 and 3, we introduce an AE-gadget for FEN-AE games for proving that such games may not admit a locally stable partition. A reduction from 3-SAT based on AE-gadgets shows the intractability of deciding the existence of locally popular or locally stable partitions. For k ∈ N \ {0}, we write
this section cite: []

Section: Simulations
We complement our theoretical study with a range of experiments.
Setup. We implemented LocPop and LocStab in Python and run the simulations on a computer with AMD Ryzen 7735HS CPU and 16GB RAM. All codes are available in the supplementary material.
For our algorithms, we chose parameters f ≤ e ∈ [0, 1] to create the friendship and enmity graphs: for two data points x and y at distance d(x, y) in an instance I, 1 we added (x, y) to the friendship graph if d(x, y) ≤ f • diam(I) and to the enmity graph if d(x, y) > e • diam(I), where diam(I) denotes the diameter of I, i.e., the maximum distance between any two points in I. We used the parameter values (f, e) ∈ {(0.2, 0.2), (0.25, 0.35), (0.4, 0.4)}. For each parameterization, we considered the appreciation-of-friends (AF), aversion-to-enemies (AE), and the balanced (B) preference domains.
this section cite: []

Section: Community Detection.
For community detection, we used four different datasets.
1. Karate club [23,31,43]: a 34-node benchmark dataset for community detection.
2. Jazz musicians [24,32]: collaboration network of 198 jazz musicians; nodes represent musicians, edges represent co-membership in a band.
3. Cora dataset [36,35]: a citation network of 2708 machine learning papers classified into seven classes; edges denote citation links.
4. Random-25: an instance containing 25 disjoint Erdős-Rényi graphs (10 nodes each, with p = 0.2) as communities; inter-community edges added independently with probability 0.05.
We compared our heuristics with two state-of-the-art algorithms, called Louvain and Leiden [2]. We tested our heuristics with respect to several different parameters. In particular, we considered the following variations for the initial clustering: (i) putting each agent into a singleton cluster (LocPop-S, LocStab-S), (ii) dividing agents randomly into k clusters where k is the predicted number of clusters (LocPop-P, LocStab-P), and (iii) using the output of the Leiden algorithm (LocPop-Ld, LocStab-Ld).
For evaluation, we used Rand index and modularity, both commonly used metrics. The Rand index assumes underlying true labels for the data points, and measures the similarity of the obtained clustering to the true clustering. More precisely, it computes the fraction of such (x, y) pairs where x and y are correctly put together or into different clusters. By contrast, modularity does not necessitate a true labeling, but instead evaluates how well the graph underlying the instance is divided into clusters, based on the number of edges within the clusters in between different clusters. See Figure 3 and Appendix C.1. Clustering. For our clustering simulations, we used four datasets.
1. Iris dataset [21]: 150 samples from three Iris species, features are sepal and petal sizes.
2. Breast cancer Wisconsin dataset [42]: 569 samples from diagnostic images (30 dimensional datapoints).
3. Moons dataset [38]: two half-moons generated by make_moons (300 points, 0.05 noise).
4. 3-Circles dataset: 300 points in three slightly overlapping circles centered at (0.5, 0.5), (0.7, 0.3), and (0.1, 0.7) with radius 0.2 and Gaussian noise (std = 0.05).
We compared our algorithms to two widely used clustering algorithms, k-means and DBSCAN, using their initial parameters. We tested our heuristic with respect to several different parameters. In particular, we considered the following variations for the initial clustering: (i) putting every agent into a singleton cluster (LocPop-S, LocStab-S), (ii) dividing the agents randomly into k clusters where k is the predicted number of clusters (LocPop-P, LocStab-P), (iii) using the output of k-means (LocPop-kM, LocStab-kM), and (iv) using the output of DBSCAN (LocPop-D, LocStab-D). To evaluate the outputs, we considered the Rand index and silhouette score, both being commonly used metrics in clustering. As opposed to Rank index, silhouette score does not assume the existence of a true labeling of the given data points, but evaluates the quality of the obtained clustering by measuring the cohesion within clusters as well as the separation between different clusters. See Figure 4 and Appendix C.2.
this section cite: ['b22', 'b30', 'b42', 'b23', 'b31', 'b35', 'b34', 'b1', 'b20', 'b41', 'b37']

Section: Key insights from experimental results.
Our algorithms run within seconds, except for Cora, where computation takes 3-5 minutes, probably due to high degrees in the friendship and enmity graphs. LocPop and LocStab performed very similarly, with LocStab depending more on the preference domain, so we discuss LocPop here. A more detailed evaluation can be found in Appendix C.3.
LocPop demonstrated strong and consistent empirical performance, performing very similarly over the three preference domains (AF, AE, and B) . We observed that parameter tuning can significantly affect outcomes, usually either the setting (0.2, 0.2) or (0.4, 0.4) performed best, and (0.25, 0.35) performed consistently between them. Interestingly, stronger performance on standard clustering metrics (silhouette score) often did not align with being closer to the true labels (Rand index).
In community detection tasks, LocPop matched or outperformed Louvain and Leiden with respect to the Rand index, particularly excelling on the Karate club (∼ 50% better) and Random-25 (∼ 10% better) datasets. The LocPop-Ld variant also matched them with respect to modularity. In clustering benchmarks, it mostly surpassed DBSCAN in both silhouette score and Rand index, and also matched or outperformed k-means (with up to 20% in 3 Circles and Iris in silhouette score). These findings indicate that LocPop is both theoretically grounded and competitive in practice, offering a flexible and efficient approach for structure discovery while providing additional robustness against deviations.
this section cite: []

Section: Conclusion
We have investigated the computational complexity of finding a popular, a locally popular, or a locally stable partition in FEN-AF, FEN-AE, and FEN-B games and identified which cases were solvable efficiently and which were NP-hard. Our algorithms can also be interpreted as dynamics of agents based on single-agent deviations and majority-based decision making.
Our experimental study provided sufficient background to show that social-choice concepts like local stability and local popularity may lead to efficient clustering and community detection techniques. Furthermore, by their nature, these concepts may often provide fairer solutions that are also more stable against potential deviations, which-as argued in the introduction-is particularly relevant in scenarios where data points correspond to agents capable of making independent actions.
this section cite: []

Section: References
Ref_id:b0 Title: Popular matchings Year: (2007)
Ref_id:b1 Title: Louvain and Leiden implementation Year: (2023-09)
Ref_id:b2 Title: Computing desirable partitions in additively separable hedonic games Year: (2013)
Ref_id:b3 Title: Hedonic games Year: (2016)
Ref_id:b4 Title: Core in a simple coalition formation game Year: (2001)
Ref_id:b5 Title: Popular matchings in the marriage and roommates problems Year: (2010)
Ref_id:b6 Title: Fast unfolding of communities in large networks Year: (2008)
Ref_id:b7 Title: The stability of hedonic coalition structures Year: (2002)
Ref_id:b8 Title: Finding and recognizing popular coalition structures Year: (2022)
Ref_id:b9 Title: Single-agent dynamics in additively separable hedonic games Year: (2022)
Ref_id:b10 Title: Economics and Computation. An Introduction to Algorithmic Game Theory Year: (2024)
Ref_id:b11 Title: Settling the complexity of popularity in additively separable and fractional hedonic games Year: (2024-11)
Ref_id:b12 Title: Stability in coalition formation games Year: (2001)
Ref_id:b13 Title: Hedonic games with friends, enemies, and neutrals: Resolving open questions and fine-grained complexity Year: (2023-06)
Ref_id:b14 Title: Popular matchings Year: (2017)
Ref_id:b15 Title: Simple priorities and core stability in hedonic games Year: (2006)
Ref_id:b16 Title: Hedonic coalitions: Optimality and stability Year: (1980)
Ref_id:b17 Title: A density-based algorithm for discovering clusters in large spatial databases with noise Year: (1996)
Ref_id:b18 Title: A comprehensive survey of clustering algorithms: State-of-the-art machine learning applications, taxonomy, challenges, and future research prospects Year: (2022)
Ref_id:b19 Title: Hedonic clustering games Year: (2015)
Ref_id:b20 Title: The use of multiple measurements in taxonomic problems Year: (1936)
Ref_id:b21 Title: Match making: assignments based on bilateral preferences Year: (1975)
Ref_id:b22 Title: Community structure in social and biological networks Year: (2002)
Ref_id:b23 Title: Community structure in jazz Year: (2003)
Ref_id:b24 Title: Popular matching in roommates setting is NP-hard Year: (2019)
Ref_id:b25 Title: Popular mixed matchings Year: (2011)
Ref_id:b26 Title: Hedonic games with ordinal preferences and thresholds Year: (2020)
Ref_id:b27 Title: Stability in FEN-hedonic games for single-player deviations Year: (2019-05)
Ref_id:b28 Title: The complexity of verifying popularity and strict popularity in altruistic hedonic games Year: (2024)
Ref_id:b29 Title: The Enron Corpus: A new dataset for email classification research Year: (2004)
Ref_id:b30 Title: Jazz musicians dataset Year: (1977)
Ref_id:b31 Title: Zachary karate club dataset Year: (2003)
Ref_id:b32 Title: Representing and solving hedonic games with ordinal preferences and thresholds Year: (2015-05)
Ref_id:b33 Title: Some methods for classification and analysis of multivariate observations Year: (1967)
Ref_id:b34 Title:  Year: (2024)
Ref_id:b35 Title: Automating the construction of internet portals with machine learning Year: (2000)
Ref_id:b36 Title: Core stability in hedonic games among friends and enemies: Impact of neutrals Year: (2017-02)
Ref_id:b37 Title: Scikit-learn: Machine learning in Python Year: (2011)
Ref_id:b38 Title: Objective criteria for the evaluation of clustering methods Year: (1971)
Ref_id:b39 Title: Borda-induced hedonic games with friends, enemies, and neutral players Year: (2018)
Ref_id:b40 Title:  Year: (2019)
Ref_id:b41 Title: Breast Cancer Wisconsin (Diagnostic) Year: (1993)
Ref_id:b42 Title: An information flow model for conflict and fission in small groups Year: (1977)
