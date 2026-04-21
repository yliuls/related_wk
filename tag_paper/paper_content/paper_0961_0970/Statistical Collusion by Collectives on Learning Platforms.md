Title: Statistical Collusion by Collectives on Learning Platforms
Abstract: As platforms increasingly rely on learning algorithms, collectives may form and seek ways to influence these platforms to align with their own interests. This can be achieved by coordinated submission of altered data. To evaluate the potential impact of such behavior, it is essential to understand the computations that collectives must perform to impact platforms in this way. In particular, collectives need to make a priori assessments of the effect of the collective before taking action, as they may face potential risks when modifying their data. Moreover they need to develop implementable coordination algorithms based on quantities that can be inferred from observed data. We develop a framework that provides a theoretical and algorithmic treatment of these issues and present experimental results in a product evaluation domain.

Section: Introduction
The dynamic interaction among agents and algorithms creates a complex ecosystem where unanticipated individual and collective behavior can emerge. The study of such behavior is crucial for understanding how to design systems that are robust, fair, and aligned with societal values.
In a network, agents often have diverse motivations and multiple incentives. When the incentives of the interacting agents do not fully align with those of the designer of the learning system, the former may wish to influence the learning process. This becomes salient in the common case in which a learning system interacts with a large number of agents. Such agents may collaborate, forming a cartel, by pooling their data and devising a common strategy. Accordingly, even if the learning algorithm is robust to single-agent Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
adversarial behavior, the collective may be able to exert a significant influence on the algorithm. This concept of collective action has its origins in economic theories (Olson, 1965) and has been more recently explored in the context of machine learning (Hardt et al., 2023). It is crucial to understand how such collectives can influence learning algorithms to better comprehend the dynamics within the network of agents and to deploy algorithms that are reliable and aligned with the majority interests of consumers. We will focus specifically on the case of a platform that deploys a learning algorithm and interacts with a population of consumers.
Collective Action in Machine Learning. We draw on the work of Hardt et al. (2023), who investigate the following problem: a population of individuals (x, y) ∈ X × Y, drawn i.i.d. from a distribution D, interacts with a platform. A collective of relative size α ∈ (0, 1) forms within this population with the goal of influencing the platform. The collective's influence is quantified by a success metric S(α), the definition of which depends on the collective's objective. To influence the platform, members of the collective modify their data according to a common strategy h : X × Y → X × Y, which maps the original features and labels of the data to modified features and labels. The platform observes a mixture of distributions α D +(1-α)D, where D is the distribution of h(z) with z ∼ D. The platform selects a classifier based on this distribution. Thus, the objective for the collective is to choose a strategy h that maximizes its success S(α). Hardt et al. (2023) propose strategies h for two distinct goals: signal planting and signal erasing. They derive lower bounds on the success S(α), enabling the identification of a minimum collective size α * such that S(α) ≥ S * for all α ≥ α * , where S * is a target threshold.
Unfortunately, several of the strategies h studied by Hardt et al. (2023) are not available in practice to the collective. Furthermore, the bounds obtained on the success S(α) depend on key parameters that are unknown to the collective. Lastly, in practice, the collective seeks a priori guarantees of success before altering its data, as modifying the data may expose members to risk. In this paper, we introduce a new framework to enable members of the collective to learn strategies h efficiently and to infer the parameters that determine their success on the platform. Contributions. Our primary contribution is the introduction of a novel framework which empowers collectives via statistical inference. This statistical framework allows for the derivation of three key results, each tied to a distinct objective that a collective may pursue to influence a platform: signal planting, signal erasing, and a new objective we introduce, signal unplanting.
We thoroughly explore strategies that a collective can employ to achieve each of these goals and provide theoretical guarantees for the effectiveness of these strategies. The statistical inference capabilities of the collective serve two main purposes: first, to estimate the most effective strategies for influencing the platform, and second, to infer key parameters that determine the collective's success. This dual approach allows the collective to predict its potential impact on the platform with high probability.
Our framework reduces to that of Hardt et al. (2023) in the infinite data regime, although we note that we improve on earlier results in this regime-our lower bounds are tighter than those from Hardt et al. (2023). Our main focus, however, is the setting in which the signal set is finite. We obtain bounds in this setting that form staircase-like curves. This result provides a new interpretation of data poisoning bounds: the success of attacks depends not just on the signal set to be poisoned as a whole but more precisely on each feature within it. Each feature has a resistance to poisoning, requiring a specific level of attack to breach it, resulting in these staircase patterns. When the steps are close enough, these curves resemble the smooth sigmoid shapes seen in data poisoning literature.
We construct a synthetic dataset to validate our theoretical findings and to examine the influence of various parameters empirically. Our empirical results highlight, among other things, that the effectiveness of the collective depends not only on its relative size compared to the total number of agents interacting with the platform but also on its absolute size. A larger collective, in absolute terms, can obtain better statistical estimates and thus more accurately infer optimal strategies. This result implies that larger platforms may be more vulnerable to collective action.
this section cite: ['b15', 'b11', 'b11', 'b11', 'b11', 'b11', 'b11']

Section: Related Work
Our research builds upon and extends the concept of collective action in machine learning, a framework originally introduced by Hardt et al. (2023). Collective action relates closely to data poisoning attacks in machine learning, a subset of security attacks that disrupt model training by injecting malicious data to degrade performance or alter predictions. Of particular virulence, backdoor attacks embed a hidden trigger in the data that activates malicious behavior only when the trigger appears, making them subtle and hard to detect. For comprehensive discussions on data poisoning, backdoor attacks, and defense mechanisms, we refer the reader to the surveys by Tian et al. (2022), Guo et al. (2021), and Cinà et al. (2023).
Data poisoning is a critical topic in machine learning. Many empirical studies focus on backdoor attacks and the defense mechanisms for learning algorithms. However, there is relatively little research that analyzes the effectiveness of these attacks theoretically. Grosse et al. (2022) show that backdoor patterns induce a stable representation of the target class. The classifier relies on the backdoor trigger and disregards other features. In the case of binary classification, Manoj & Blum (2021) demonstrate that if the model has a property called nonzero memorization capacity, then a successful backdoor attack is possible. The model's vulnerability is assessed based on its ability to memorize out-ofdistribution values. In particular, overparameterized linear models have higher memorization capacity and are more susceptible to attacks. Xian et al. (2023) also investigate the context of binary classification and propose a hypothesis regarding the distribution of poisoned data, which allows them to derive useful results on the effectiveness of an attack. Wang et al. (2024) explore the effectiveness of backdoor attacks from a statistical standpoint. They provide bounds on the statistical risks associated with a poisoned model, specifically analyzing how these risks manifest when the model is evaluated on both clean and backdoored data for a finite sample size. Li & Liu (2024) present a theoretical examination of a backdoor attack applied to a convolutional neural network with two layers. Cinà et al. (2024) conduct an empirical study on the learning curves associated with backdoor attacks. Moreover, they demonstrated that classifiers with stronger regularization are generally more resistant to poisoning attacks, although this comes with a slight decrease in accuracy on clean data.
What sets the concept of collective action apart conceptually is its treatment of the collective as a group of individuals, each representing a data point. This perspective also has economic, social, and political dimensions, as certain groups of individuals can unite and collaborate to influence decisions. Such ideas have been explored in areas of research at the intersection of machine learning and other fields (Vincent et al., 2019;Albert et al., 2020;Vincent et al., 2021;Albert et al., 2021;Creager & Zemel, 2021;Vincent & Hecht, 2021). See Appendix A of Hardt et al. (2023) for an indepth analysis of the related work on collective action. In addition, Ben-Dov et al. (2024) highlight how the learning algorithm shapes the success of collective action.
An important contribution of Hardt et al. (2023) is to study data poisoning via the formalism of Bayes-optimal classification, which yields a conceptual inversion of the idea of strategic classification (Hardt et al., 2016). While strategic classification revolves around a firm's ability to anticipate and respond to the actions of a single, strategic individual, collective action shifts the focus toward a scenario where individuals collectively anticipate and strategically respond to the optimizing behavior of the firm. This concept has also been explored by Zrnic et al. (2021). Unlike traditional strategic classification, which primarily considers the firm's perspective, collective action highlights the role of workers and consumers on online platforms.
this section cite: ['b11', 'b9', 'b7', 'b14', 'b21', 'b20', 'b13', 'b5', 'b18', 'b0', 'b19', 'b1', 'b6', 'b19', 'b11', 'b2', 'b11', 'b10', 'b22']

Section: Statistical Algorithmic Collective Action in Classification
First, we will describe the new setting for deriving theoretical bounds in collective action that are effectively computable by the collective. We then present our three main results, which address three different objectives for the collective. Two of these objectives are classic goals of collective action: signal planting and signal erasing. Additionally, we introduce a new objective: signal unplanting, where the collective aims to prevent an association between the signal set and a certain label. We will explore strategies and provide theoretical guarantees for each of these objectives.
this section cite: []

Section: Setting
We consider a platform that deploys a learning algorithm in a universe X × Y. We assume that X × Y is finite. Each individual corresponds to a single data point (x, y) ∈ X ×Y.
The platform trains a classifier f on a training dataset. The training dataset is composed of N consumers which are initially drawn i.i.d. according to some distribution D. Among these consumers, a certain number n < N forms a collective to strategically influence the firm's behavior. The collective shares a common strategy h : X × Y → X × Y. The N -n base consumers and the n members of the collective together form an empirical distribution of consumers P, and the corresponding dataset constitutes the training set. The collective wants to obtain guarantees on the influence they have on the platform at test time.
Notation. Given a distribution Q over X × Y, we denote by Q X the marginal distribution over features. We will simply write Q when the context allows. We denote by D the distribution of h(z), z ∼ D. More generally, for a dataset D, we write D := {h(z) | z ∈ D} (as a multiset) for the same dataset after applying strategy h. We will also write, for E ⊆ X × Y, P z∼D (z ∈ E) := 1 #D zi∈D 1 {zi∈E} the empirical probability of the event E induced by a dataset D. We use P to denote empirical probabilities, and P for population probabilities.
When the variables do not need to be explicitly stated, we may write P D (E) and P D (E) instead of P z∼D (z ∈ E) and P z∼D (z ∈ E) respectively. The collective. Given a test set of consumers D test i.i.d.
∼ D, the collective's goal is to obtain guarantees with high probability on their success Ŝ(n) as a function of D test . The definition of Ŝ(n) is based on the objective desired by the collective. We will consider three objectives: signal planting, signal unplanting, and signal erasing. The collective modifies its data using strategy h to maximize its success.
We assume that the collective has access to the value N . This is a weak assumption because it is common in practice. For example, if the platform is a polling institute seeking to understand participants' voting preferences based on their demographic data, the total number of people surveyed is usually publicly available.
Unless stated otherwise, the collective has access only to their own data and not to the data of consumers who are not part of the collective. The collective can pool its data to infer quantities that depend on the underlying distribution. Throughout this paper, we will use Hoeffding's concentration inequality (Lemma D.1) for simplicity. We will denote Hoeffding error terms as follows:
R δ (k) := log(1/δ) 2k , for any δ > 0 and k ∈ N * . We note that Hoeffding's inequality can be loose, for example when applied to sums of Bernoulli random variables with means close to zero. One can address this issue by using other concentration inequalities such as Bernstein; the framework we present here can readily incorporate such choices.
The platform. The firm observes an empirical distribution of consumers P. It selects a classifier f based on this distribution P. Following Hardt et al. (2023), we characterize classifiers f by their suboptimality in terms of total variation distance with respect to the observed distribution P:
Definition 3.1. Let ε > 0. A classifier f : X → Y is ε-suboptimal on a set X ′ ⊆ X
under the distribution P if there exists a distribution P with T V ( P, P) ≤ ε such that f (x) ∈ argmax y∈Y P(x, y) for all x ∈ X ′ . Here, T V ( P, P) := sup E⊆X ×Y | P(E)-P(E)|
denotes the total variation distance between P and P.
The parameter ε roughly controls how much the classifier can make use of statistics that go beyond simple frequency counts. It accounts for classifiers that consider feature interactions and capture complex patterns in the data.
this section cite: ['b11']

Section: Signal planting
In signal planting, we are given a transformation g : X → X and a target label y * ∈ Y. The map g induces a signal set defined by X := {g(x) | x ∈ X }. The success is defined as
Ŝ(n) := P x∼Dtest ( f (g(x)) = y * ),
where
D test i.i.d.
∼ D is the test set. In other words, the collective aims to enforce an association between the signal set X and a target label y * at test time.
this section cite: []

Section: FEATURE-LABEL SIGNAL PLANTING
A natural goal for the collective is to maximize its success, Ŝ(n). To ensure that its efforts are effective, the collective aims to establish theoretical lower bounds on Ŝ(n), as these provide a guarantee of success. To do so, the collective can play the feature-label signal planting strategy defined below.
Definition 3.2 (Feature-label signal planting strategy). We define the feature-label signal planting strategy as
h(x, y) = (g(x), y * ).
We analyze the effect of this strategy on the learning platform. Formally, we are given three independent datasets:
a dataset D (n) i.i.d. ∼ D of n consumers which are part of the collective; a dataset D (N -n) i.i.d.
∼ D of N -n consumers which are not part of the collective; and a dataset
D test i.i.d. ∼ D of N test consumers forming the test dataset. We recall that D(n) := {h(z) | z ∈ D (n) } (as a multiset).
The training set of the platform's classifier is the concatenation of D(n) and D (N -n) . In other words, the platform observes the distribution:
P(x 0 , y 0 ) := n N P D(n) (x 0 , y 0 ) + N -n N P D (N -n) (x 0 , y 0 ).
The platform then chooses a classifier f based on P. We can now state the main result for signal planting:
Theorem 3.3 (Signal planting lower bound, feature-label signal planting strategy). Let δ > 0, and write δ := δ/(2 + 2# X +2# X #Y). Then, by playing the feature-label signal planting strategy against a classifier that is ε-suboptimal on X , the collective achieves with probability at least 1 -δ (over the draw of the consumers):
Ŝ(n) ≥ P x∼ D(n) n N P D(n) (x) -2R δ (n) - N -n N ∆ (n) x + 2R δ (n) + 2R δ (N -n) - ε 1 -ε > 0 -R δ (n) -R δ (N test ),(1)
where
∆ (n) x := max y ′ ∈Y\{y * } P D (n) (x, y ′ ) - P D (n) (x, y * ).
Proofs and additional remarks can be found in Appendix E. Note that lower bound (1) is fully computable by the collective as it depends only on datasets D (n) and D(n) . D (n)   is obtained through data pooling by the collective's members, and D(n) is computed by applying strategy h to the dataset D (n) . Details on the algorithm for computing the lower bound are provided in Appendix B, and the code is available at: https://github.com/GauthierE/   statistical-collusion.
The interpretation of lower bound (1) is that success increases step by step: each feature x in the signal set X has a certain resistance to being planted. As the relative size of the collective n/N gradually increases, features x are cracked as their resistance breaks, in decreasing order of resistance. For each feature x, its resistance depends on three terms. The first one, here n N ( P
D(n) (x) -2R δ (n))
, represents how prevalent the feature x is in the modified data: the more frequently x appears in the poisoned data, the greater the collective's ability to influence the associated label. The second term, here
-N -n N (∆ (n) x + 2R δ (n) + 2R δ (N -n))
, captures the counteracting influence of non-collective individuals in the population, quantifying how much they might limit the collective's success in planting the signal. It indicates how strongly the target label is associated with the signal set: the more frequent y * is in the signal set, the easier it is to plant the signal; if other labels are far more likely than y * , planting the signal becomes more difficult. The third term, -ε 1-ε , represents the platform's ability to adapt to the distribution of its users. As ε → ε 1-ε increases with ε, it benefits the collective to have ε close to zero, limiting the platform's flexibility and resulting in a tighter bound.
The first term scales approximately linearly with n, by a factor of n/N , while the second term decreases by 1 -n/N . However, the dependence is more complex than purely linear. The bound involves estimation terms R δ (n), which decay at a rate proportional to n -1/2 as n increases. The bound also depends on R δ (N -n), but these terms can be negligible as long as n remains much smaller than N .
Also, the cardinality of X affects the definition of δ, and thus the estimation terms R δ . The smaller # X is, the better the collective's estimates will be, resulting in sharper bounds.
this section cite: []

Section: FEATURE-ONLY SIGNAL PLANTING
Note that the feature-label signal planting strategy assumes that the members of the collective can change both their features and their labels. This might not always be feasible in practice, where labels can be immutable. In this situation, a natural strategy for the collective is to change its feature x to g(x) when y = y * . This strategy was explored by Hardt et al. (2023). However, it is preferable for the collective to additionally change its feature x to some x 0 that is not in the signal set X when y ̸ = y * . This ensures that not only does the feature belong to the signal set X when the label is y * , but also that if a feature is in the signal set X , the associated label is necessarily y * . This dual condition strengthens the association between the signal set X and the target label y * . In this case, we can directly say that the term
P D(n) (g(x ′ ), y ′ )
with y ′ ̸ = y * that would appear in the proof of Theorem 3.5 is equal to zero, leading to a sharper bound.
Definition 3.4 (Feature-only signal planting strategy). We define the feature-only signal planting strategy as h(x, y) = (g(x), y * ) if y = y * , (x 0 , y) otherwise, where x 0 ∈ X \ X is any feature that does not belong to the signal set X . Note that x 0 does not have to be fixed across all initial data points (x, y).
We can generalize Theorem 3.3 to the feature-only strategy:
Theorem 3.5 (Signal planting lower bound, feature-only signal planting strategy). Let δ > 0, and write δ := δ/(2 + 2# X + 2# X #Y). Then, by playing the feature-only signal planting strategy against a classifier that is ε-suboptimal on X , the collective achieves with probability at least 1 -δ (over the draw of the consumers):
Ŝ(n) ≥ P x ′ ∼D (n) n N P D(n) (g(x ′ ), y * ) -2R δ (n) - N -n N ∆ (n) g(x ′ ) + 2R δ (n) + 2R δ (N -n) - ε 1 -ε > 0 -R δ (n) -R δ (N test ),(2)
where
∆ (n) g(x ′ )
is defined in Theorem 3.3. We can straightforwardly compare lower bounds (1) and ( 2). The only difference is that in (2), we have
P D(n) (g(x ′ ), y * ) instead of P D(n) (g(x ′
)). This comparison directly measures the impact of not modifying labels on the collective's influence. In the following, we will always assume that the collective can modify its labels.
Signal planting relies on a straightforward strategy: flooding the platform with as many pairs (g(x), y * ) as possible. This strategy is simple and does not require the collective to perform any statistical estimation. Statistical inference is only necessary for the collective to calculate the lower bound on success, not for defining the optimal strategy h. In the next two sections, we examine two objectives that additionally require statistical estimation to infer the optimal strategy h: signal unplanting and signal erasing.
this section cite: ['b11']

Section: Signal unplanting
The setting is essentially the same as before; the only difference is the success of the collective is defined as follows:
Ŝ(n) := P x∼Dtest f (g(x)) ̸ = y * .
The collective's goal is now to prevent an association between the signal set X = {g(x) | x ∈ X } and the target label y * .
this section cite: []

Section: NAIVE STRATEGY
A simple and naive strategy for the collective is to flood the platform with feature-label pairs of the form (g(x), y ′ ) for some fixed y ′ ̸ = y * . Indeed, for any y ′ ̸ = y * , we have that
Ŝ(n) = P x∼Dtest f (g(x)) ̸ = y * ≥ P x∼Dtest f (g(x)) = y ′ =: Ŝy ′ (n).
Therefore, the collective can compute lower bounds on Ŝy ′ (n) for all y ′ ̸ = y * using Theorem 3.3. It can then play the strategy h(x, y) = (g(x), ȳ) where ȳ is the label that maximizes the lower bounds on Ŝy ′ (n) for y ′ ̸ = y * . Note that this is equivalent to planting a signal with transformation g and target label ȳ, so we obtain the same guarantees as in Theorem 3.3.
this section cite: []

Section: ADAPTIVE STRATEGY
The collective can also use an adaptive strategy, meaning that each member of the collective (x, y) can change its feature-label pair to some (g(x), y g(x) ) where the modified label y g(x) depends on g(x) and is no longer fixed. The plan for the collective is to estimate the optimal label using a subset of n e < n randomly chosen participants. Formally, we assume that D (n) is the concatenation of two independent datasets D (ne) and D (n-ne) drawn from D. A natural strategy for the collective is to change every pair (x, y) into (g(x), ŷg(x) ) where for x ∈ X :
ŷx := argmax y ′ ∈Y\{y * } P D (ne ) (x, y ′ ).(3)
We formalize this strategy in the following definition.
Definition 3.6 (Signal unplanting strategy). We define the signal unplanting strategy as
h(x, y) = (g(x), ŷg(x) ),
where ŷx is defined in Equation (3).
Intuitively, this strategy means that the collective aims to select the most likely label among all labels different from y * , given a feature g(x).
Theorem 3.7 (Signal unplanting lower bound). Let δ > 0, and write δ := δ/(2 + 6# X ). Let n e < n. Then, by playing the signal unplanting strategy above against a classifier that is ε-suboptimal on X , the collective achieves with probability at least 1 -δ (over the draw of the consumers):
Ŝ(n) ≥ P x∼ D(n) n N P D(n) (x) -2R δ (n) - N -n N ∆ (n-ne) x + 2R δ (n -n e ) + 2R δ (N -n) - ε 1 -ε > 0 -R δ (n) -R δ (N test ),(4)
where
∆ (n-ne) x := P D (n-ne) (x, y * ) - P D (n-ne)
(x, ŷx ).
this section cite: []

Section: Signal erasing
In the previous subsections, we examined how the collective can plant a signal by using a natural strategy, which consists in simply flooding the platform with feature-label pairs of the form h(x, y) = (g(x), y * ), and how it can unplant a signal by estimating the most probable label different from y * . In this section, we study another objective: signal erasing.
We consider a transformation g : X → X . The success of the collective is now defined by:
Ŝ(n) := P x∼Dtest f (g(x)) = f (x) .
As outlined by Hardt et al. (2023), maximizing Ŝ(n) aligns with reducing the impact of g on the learning algorithm. The term signal erasing is motivated by the example in tabular data where g preserves certain features while removing others, for instance by setting some features to a fixed value. This effectively removes the impact of the erased features, provided these features are independent of the other ones.
In this part, we will make the following mild assumptions.
Assumption (A1): ∃η > 0, ∀x ∈ X , ∃y * x ∈ Y : ∀y ′ ̸ = y * x, P D (x, y * x) > P D (x, y ′ ) + η.
Intuitively, Assumption 1 implies two things. Firstly, each feature x ∈ X is sufficiently frequent in the base distribution. Secondly, given a feature x ∈ X , there exists a label y *
x that is consequently more probable than the other labels in the base distribution.
Assumption (A2): The transformation g is idempotent: g(g(x)) = g(x) for all x ∈ X .
To understand Assumption 2, consider data poisoning in image classification. Assumption 2 holds in data poisoning strategies where the trigger is a fixed, opaque watermark, as in the case of binary masks (Gu et al., 2019), where applying the mask twice is equivalent to applying it once. However, this is not true for strategies like pixel blending (Chen et al., 2017). In contrast, Assumption 2 naturally applies to tabular data, where the collective can apply a transformation g to map a feature to a constant value. Now, we outline a scheme that the collective can use to compute a lower bound. The basic idea is that each member (x, y) of the collective keeps its feature x but changes its label to the most likely label y * g(x) based on the feature g(x). This approach encourages the platform to predict the same label for both x and g(x), hence erasing the signal. The scheme is the following: first, the collective pools its data to predict the optimal y *
x for each x ∈ X . Then, the collective applies some strategy h based on the first step to erase the signal. Assuming the collective can compute the optimal label y *
x for each x ∈ X , the erasure strategy is formally defined as follows:
Definition 3.8 (Erasure strategy). We define the erasure strategy as h(x, y) = (x, y * g(x) ), where y *
x is given in Assumption 1 for each x ∈ X . We can now state the main result in signal erasing: Theorem 3.9 (Signal erasing lower bound). Let δ > 0, and write δ := δ/(2
+ # X #Y + 2#X + 2#X #Y). Assume that 2 log(1/ δ) η 2 ≤ n ≤ N -2 log(1/ δ) η 2
where η is given in Assumption 1. Then, with probability at least 1 -δ (over the draw of the consumers), the collective can compute y *
x for all x ∈ X and by playing the erasure strategy it achieves against a classifier that is ε-suboptimal on X :
Ŝ(n) ≥ P x ′ ∼D (n) n N P D (n) (x ′ ) -2R δ (n) - N -n N ∆ (n) x ′ + 2R δ (n) + 2R δ (N -n) - ε 1 -ε > 0 -R δ (n) -R δ (N test ),(5)
where ∆
(n)
x ′ := max y ′ ∈Y\{y * g(x ′ ) } P D (n) (x ′ , y ′ ) - P D (n) (x ′ , y * g(x ′ ) ).
In signal erasing, just like in signal unplanting, the collective leverages its own data to compute the strategy h. However, the technique differs. In signal unplanting, the collective uses a fraction of its members to estimate the optimal label to play. Whereas in signal erasing, the collective, provided that it is sufficiently large, utilizes all of its data to compute the most likely label given a feature x, which exists under Assumption 1.
this section cite: ['b11', 'b8', 'b3']

Section: Experimental Evaluation
In our experiments, we simulate a platform that collects data on vehicles, where each sample represents a car. The features of each car include characteristics such as Model Type, Fuel Type, and Country of Manufacture. The labels assigned to each vehicle reflect the car's evaluation, categorized into four classes: Excellent, Good, Average, or Poor. Further details on the dataset and the specific parameters used in the experiments can be found in Appendix A.
We consider a scenario where a collective seeks to influence the platform by lobbying against a particular category of vehicles, specifically SUVs with specific features. The collective defines a signal set X through a transformation g fixing all feature values except Country of Manufacture.
They may want to plant a signal targeting a label y * = Poor. They might also aim to unplant signals, specifically working to associate elements of X with labels y ̸ = Excellent.
this section cite: []

Section: Signal planting
In Figure 1, we plot the lower bounds from Theorem 3.3 for various values of n and compare them to the true success Ŝ(n) observed at test time. We fit sigmoid functions to interpolate the obtained values. The lower bounds are indeed lower than the success, and the gap between the two is not excessively large. This gap could potentially be narrowed using more advanced statistical inference methods. Interestingly, even though in practice the label y * = Excellent is already the most frequent label for every element of the signal set X and does not technically need to be planted-i.e., Ŝ(n) = 1 for all n-the collective is still not guaranteed to have influence over the signal for small values of n. This stems from statistical uncertainties, which our framework highlights, causing the collective to consistently overestimate the number of agents needed for a given success.
Figure 2 shows how the lower bound evolves with the total number of individuals N . When the fraction n/N is held fixed, increasing N leads to a larger collective size n. This, in turn, improves the collective's ability to estimate key quantities, as reflected by the decreasing error terms R δ (n). This suggests that platforms interacting with large user bases are more exposed to collectives altering their data. While this observation is specific to signal planting, it is even more relevant when optimal strategies need to be estimated, as in signal unplanting.
this section cite: []

Section: Signal unplanting
We now focus on signal unplanting with a target y ̸ = Excellent, where the collective aims for the platform to predict a label other than Excellent for samples in the signal set X .
The lower bound obtained in Theorem 3.7 depends on n e , the size of the sub-collective used to determine the strategy h. As shown in Figure 3 (a), n e involves a trade-off: small values lead to erratic strategy estimates and weaker bounds, while overly large values increase the R δ (n -n e ) term, also weakening the bound. A good balance is achieved at n e = 2, 000.
Figure 3 (b) compares this adaptive strategy with n e = 2, 000 to naive strategies planting labels y * ∈ {Good, Average, Poor}. The adaptive strategy consistently outperforms the naive ones by providing a higher lower bound, demonstrating the benefits of tailoring the strategy based on the data. However, it is worth noting that the naive strategy of planting the signal with y * = Good performs well for this dataset. This is due to the fact that most elements in the signal set X have Good as the second most likely label after Excellent.
In Figure 3 (c), we compare the lower bound with n e = 2, 000 and the actual success Ŝ(n) achieved at test time. 0 5 10 15 20 25 30 0.0 0.2 0.4 0.6 0.8 1.0 (a) 4 6 8 10 12 14 16 0.0 0.2 0.4 0.6 0.8 1.0 100 2,000 10,000 0 5 10 15 20 25 30 0.0 0.2 0.4 0.6 0.8 1.0 (b) 4 6 8 10 12 14 16 0.0 0.2 0.4 0.6 0.8 1.0 2,000 G A P 0 5 10 15 20 25 30 0.0 0.2 0.4 0.6 0.8 1.0 (c) 2,000 Ŝ(n)
this section cite: []

Section: Success
Relative collective size n/N (in %)
Zoomed-in view The lower bound suggests that the collective would need to represent around 10% of the agents interacting with the platform to have a significant impact, while in practice, only about 3% was sufficient. Our finite-sample framework differs from that presented by Hardt et al. (2023), which focuses on a population-level analysis. The connection is that the limit of our framework in the infinite data regime boils down to that of Hardt et al. (2023). Specifically, for α ∈ (0, 1). let n, N, N test → ∞ such that n/N → α. In this case, our statistical algorithmic collective action framework simplifies to that of algorithmic collective action. In Figure 4, we compare the bounds we obtain under the infinite data regime with those of Hardt et al. (2023). The bounds presented in our work are tighter than those obtained by Hardt et al. (2023). We provide a proof and additional remarks in Appendix F.
this section cite: ['b11', 'b11', 'b11', 'b11']

Section: Comparative analysis
Interestingly, in the case of a discrete signal set, as used in our experiments, our lower bounds take on a staircase shape rather than a smooth sigmoid. This mirrors the shape of the curves observed in signal unplanting in Figure 3, which would also appear in Figure 1 and Figure 2 if we used finer increments of n. In contrast, the bounds from Hardt et al. (2023) do not capture this phenomenon.
this section cite: ['b11']

Section: Discussion
In this work, we introduced a framework where collectives aiming to influence a platform can leverage their local information by pooling their data. Our approach captures the key desideratum that collectives may not only want to observe the outcomes of their actions at test time but also anticipate lower bounds on their success. Moreover, it allows the collective to implement practical strategies based on parameters they do not directly observe, by using statistical estimation. The ability to anticipate outcomes and make informed decisions based on pooled data represents an advancement in understanding how collectives can interact with and influence platforms. However, there is room for further exploration.
We used Hoeffding's inequality as a proof of concept, but it is worth studying improved concentration inequalities for obtaining estimates of success for collectives. It is also possible to reverse the concentration inequalities to derive upper bounds on success, but the challenge lies in the fact that these upper bounds are often trivial. In ongoing work, we are exploiting recent developments in concentration inequalities due to Howard et al. (2020).
We focused on classification, but extensions to other objectives, such as regression, would be useful. We also assumed that the feature space X is finite to use a limited number of events in the union bounds. It would be natural to remove this assumption by using a covering of X , provided it is compact, to derive more general bounds for signal planting and unplanting.
Finally, a key assumption in collective action is that individuals are identically distributed. However, those who join a collective are often distributed differently from the general population-e.g., a collective against SUVs is likely to have fewer SUV-related data points and more data on smaller vehicles. Extending the framework to account for a heterogeneous population is an important direction for future work.
this section cite: ['b12']

Section: References
Ref_id:b0 Title: Politics of adversarial machine learning Year: (2020)
Ref_id:b1 Title: Adversarial for good? How the adversarial ML community's values impede socially beneficial uses of attacks Year: (2021)
Ref_id:b2 Title: The role of learning algorithms in collective action Year: (2024)
Ref_id:b3 Title: Targeted backdoor attacks on deep learning systems using data poisoning Year: (2017)
Ref_id:b4 Title: Wild patterns reloaded: A survey of machine learning security against training data poisoning Year: ()
Ref_id:b5 Title: Backdoor learning curves: Explaining backdoor poisoning beyond influence functions Year: (2024)
Ref_id:b6 Title: Online algorithmic recourse by collective action Year: (2021)
Ref_id:b7 Title: Backdoor smoothing: Demystifying backdoor attacks on deep neural networks Year: (2022)
Ref_id:b8 Title: Evaluating backdooring attacks on deep neural networks Year: (2019)
Ref_id:b9 Title: An overview of backdoor attacks against deep neural networks and possible defences Year: (2021)
Ref_id:b10 Title: Strategic classification Year: (2016)
Ref_id:b11 Title: Algorithmic collective action in machine learning Year: (2023)
Ref_id:b12 Title: Time-uniform Chernoff bounds via nonnegative supermartingales Year: (2020)
Ref_id:b13 Title: A theoretical analysis of backdoor poisoning attacks in convolutional neural networks Year: (2024)
Ref_id:b14 Title: Excess capacity and backdoor poisoning Year: (2021)
Ref_id:b15 Title: The logic of collective action: Public goods and the theory of groups Year: (1965)
Ref_id:b16 Title: A comprehensive survey on poisoning attacks and countermeasures in machine learning Year: ()
Ref_id:b17 Title: Can "conscious data contribution" help users to exert "data leverage" against technology companies? Year: ()
Ref_id:b18 Title: Data strikes": Evaluating the effectiveness of a new form of collective action against technology companies Year: (2019)
Ref_id:b19 Title: Data leverage: A framework for empowering the public in its relationship with technology companies Year: (2021)
Ref_id:b20 Title: Demystifying poisoning backdoor attacks from a statistical perspective Year: (2024)
Ref_id:b21 Title: Understanding backdoor attacks through the adaptability hypothesis Year: (2023)
Ref_id:b22 Title: Who leads and who follows in strategic classification? Year: (2021)
