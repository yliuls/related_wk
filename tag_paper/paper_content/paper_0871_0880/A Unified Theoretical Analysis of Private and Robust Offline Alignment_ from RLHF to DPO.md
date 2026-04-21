Title: A Unified Theoretical Analysis of Private and Robust Offline Alignment: from RLHF to DPO
Abstract: In this paper, we theoretically investigate the effects of noisy labels in offline alignment, with a focus on the interplay between privacy and robustness against adversarial corruption. Specifically, under linear modeling assumptions, we present a unified analysis covering both reinforcement learning from human feedback (RLHF) and direct preference optimization (DPO) under different privacy-corruption scenarios, such as Local differential privacy-then-Corruption (LTC), where human preference labels are privatized before being corrupted by an adversary, and Corruption-then-Local differential privacy (CTL), where labels are corrupted before privacy protection. Our analysis leverages a reduction framework that reduces the offline alignment problem under linear modeling assumptions to parameter estimation in logistic regression. This framework allows us to establish an interesting separation result between LTC and CTL, demonstrating that LTC presents a greater challenge than CTL in offline alignment, even under linear models. As important by-products, our findings also advance the state-of-the-art theoretical results in offline alignment under privacy-only or corruption-only scenarios. work can simultaneously handle three privacy-corruption scenarios for both RLHF and DPO: Corruption-then-LDP (CTL), LDP-then-Corruption (LTC), and Corruption-LDP-Corruption (CLC), capturing different ways privacy and corruption may interact in practice. Reduction to Logistic Regression.Our unified analytical framework leverages a reduction that transforms the offline alignment problem, under certain linear modeling assumptions, into parameter estimation in logistic regression. This reduction enables us to establish suboptimality bounds for both RLHF and DPO by focusing on parameter estimation in logistic regression under private and corrupted labels across different scenarios. Moreover, it highlights key differences between RLHF and DPO, providing insights into practical design considerations. Separation between CTL and LTC.A key takeaway from our study of the interplay between privacy and robustness to corruption is that LTC is a more challenging setting than CTL, illustrating that the order in which privacy and corruption interact with each other significantly impacts the performance of offline alignment. New State-of-the-art Guarantees.Our results, when reduced to privacy-only or corruption-only settings, set new state-of-the-art results on theoretical guarantees for RLHF and DPO. For instance, for DPO under "corrupted" labels, our result is the first one that achieves O(1/ √ n) rate (where n is the size of preference dataset), matching the standard rate without noise. Additionally, as a by-product of our reduction approach, we provide the first results on parameter estimation error in logistic regression under both private and corrupted labels, which may be of independent interest.

Section: Introduction
The alignment training process in language models that utilizes a human-labeled preference dataset has been instrumental in producing more helpful, harmless, and honest responses (Bai et al., 2022). Leveraging an offline preference dataset, two prominent paradigms have emerged. The first is the indirect approach, such as Reinforcement Learning from Human Feedback (RLHF) (Ziegler et al., 2019;Ouyang et al., 2022), which learns an intermediate reward model before optimizing the policy. The second is the direct approach, exemplified by Direct Preference Optimization (DPO) (Rafailov et al., 2023), which directly optimizes the policy via supervised learning on the preference dataset.
It is clear that the performance of both RLHF and DPO is significantly influenced by the quality of the preference labels in the dataset. However, in practice, these labels are often noisy due to various factors (Lambert et al., 2023). One potential noise source is corruption or misspecification during label generation or data collection, e.g., data poisoning attack (Casper et al., 2023). Additionally, privacy concerns in human preference (as illustrated in Feng et al. (2024)) may prompt individuals to provide noisy or privatized preferences rather than their true rankings.
From a theoretical perspective, understanding the impact of these noisy labels-resulting from both corruption and privacy-is essential for improving offline alignment. Recent studies have made some initial attempts to address this issue (Mandal et al., 2024;Chowdhury et al., 2024;2023;Bukharin et al., 2024), but they face two fundamental limitations: (1) They often treat corruption and privacy separately and focus exclusively on either RLHF or DPO, while, in practice, noisy labels can stem from both factors simultaneously; (2) The theoretical guarantees provided by these studies are often suboptimal, even when privacy and corruption are separately considered. Motivated by these limitations and practical scenarios, we are particularly interested in the following question:
Can we provide a unified analysis of the interplay between privacy and robustness in both RLHF and DPO?
We provide an affirmative answer to the above question by presenting the following contributions:
1. A Unified Theoretical Framework. We present a unified theoretical framework for analyzing the interplay between privacy and robustness in offline alignment, covering both RLHF and DPO. Specifically, for privacy protection, we consider Local Differential Privacy (LDP) (Kasiviswanathan et al., 2011;Duchi et al., 2013) for preference labels, while for robustness, we consider the strong adversary corruption model (Diakonikolas & Kane, 2023), where an adaptively chosen fraction of labels can be corrupted. Our frame-Finally, we remark that, as in many previous related works, e.g., Zhu et al. (2023); Chowdhury et al. (2023), we consider linear modeling assumptions for the sake of theoretical analysis. However, we believe that our results could serve as important benchmarks for more general function classes.
In fact, we have also verified our separation result between CTL and LTC in the general case via experiments on GPT2large, see Appendix D for a detailed discussion.
this section cite: ['b2', 'b49', 'b32', 'b35', 'b27', 'b5', 'b20', 'b30', 'b14', 'b4', 'b26', 'b16', 'b15', 'b48', 'b13']

Section: Related Work
In the main body, we only focus on the most related work on robust and private offline alignment, while relegating an additional discussion to Appendix A.
Provably robust alignment under corruption. Mandal et al. (2024) considers offline RLHF with corrupted preference datasets and establishes upper bounds on the suboptimality gap under various coverage assumptions of the offline dataset. As will be discussed in Section 6.1, their results are either suboptimal or lack rigor due to gaps in their proof. For robust DPO, Chowdhury et al. (2024) considers a strictly weaker corruption model and derives a suboptimality bound of rate O(1/n 1/4 ). In contrast, our general result, when reduced to the same corruption model, achieves a better rate of O(1/ √ n). Bukharin et al. (2024) also considers a specific corruption model in the label generation process of RLHF but only provides the estimation error of the reward model, without a performance guarantee for the final policy.
this section cite: ['b30', 'b14', 'b4']

Section: Provably Private Alignment.
The most related work in this aspect is Chowdhury et al. (2023), which mainly focuses on the reward model estimation in RLHF under various privacy constraints (i.e., local and central label differential privacy). Our intermediate result on estimation error (Section 5) recovers the one in Chowdhury et al. (2023) when the corruption parameter is set to zero. Moreover, compared to the implicit suboptimality bound in Chowdhury et al. (2023), we provide the first explicit bound in terms of the relative condition number (Agarwal et al., 2021), which parallels similar results in standard (robust) offline RL (Zhang et al., 2022), i.e., reward-based rather than preference-based.
this section cite: ['b13', 'b13', 'b13', 'b0', 'b46']

Section: Preliminaries
Background on Offline Alignment. The goal of offline alignment is to further tune the Supervised Fine-Tuning (SFT) model to match human preferences using an offline preference dataset. The preference dataset D = (s i , a 0 i , a 1 i , y i ) n i=1 consists of n samples, each has one context/state s i (e.g., prompt), two actions a 0 i , a 1 i (e.g., two answers from language models) and label/preference feedback y i ∈ {0, 1} indicating which one is preferred by humans. We assume s i to be sampled independently from a distribution ρ. A widely used approach for modeling y i is Bradley-Terry model (Bradley & Terry, 1952):
P y i = l|s i , a 0 i , a 1 i = exp(r ⋆ (si,a l i )) exp(r ⋆ (si,a 0 i ))+exp(r ⋆ (si,a 1 i )) , (1
)
for l ∈ {0, 1}, where r ⋆ (•, •) is a ground truth reward model.
Based on this preference dataset, offline alignment aims to learn a good policy π. In particular, the performance of the learned policy π is evaluated by the suboptimality gap between π and a comparator policy π † , defined as
SubOpt( π, π † ) = J(π † ) -J( π),(2)
where J(π) := E s∼ρ,a∼π(•|s) [r ⋆ (s, a)] and π † is not necessarily the optimal policy.
RLHF and DPO. As already mentioned, there are two major paradigms in alignment for finding π: indirect and direct approaches. The former, exemplified by RLHF (Ziegler et al., 2019), involves an intermediate reward model learning process from preference dataset D before the policy optimization. The latter, represented by DPO (Rafailov et al., 2023), employs a direct policy optimization, i.e., using a supervised-learning loss function to optimize the policy directly over the preference dataset D.
this section cite: ['b3', 'b49', 'b35']

Section: Privacy Protection in Human Feedback.
The preference signal y i in D could reveal sensitive personal information (Feng et al., 2024;Chowdhury et al., 2023), hence requiring a rigorous privacy protection. To this end, we consider the local label Differential Privacy (DP) (Chaudhuri & Hsu, 2011;Ghazi et al., 2021), which means that the learner now only has access to a privatized label rather than the raw one. More specifically, we have the following definition.
Definition 3.1 (Label DP in Local Model (Chowdhury et al., 2023)). Let ε > 0 and δ ∈ [0, 1]. If each label is privatized by a local randomizer R, which satisfies for any y, y ′ and any subset S in the range of R that
P{R(y) ∈ S} ≤ e ε • P{R (y ′ ) ∈ S} + δ,
then we say R is an (ε, δ)-label differentially private local randomizer, and this privatized dataset is called label-private preference dataset. The entire alignment process that operates with the privatized dataset is said to satisfy local label DP. When δ = 0, we simply say it is a ε-local label DP.
Remark 3.2 (Randomized Response). Given the binary data of the true label, we would like to maintain the binary data property after privatization. Thus, we will adopt the standard randomized response mechanism (Warner, 1965) as our local randomizer, which essentially injects controllable noise in labels by a random flipping. Here, by "controllable," we mean the noise injection method, and noise level is under our control based on the privacy parameter ε.
Corruption in Human Feedback. The human feedback y i can often be noisy and even be corrupted in the source or during the data collection process, which deviates from the assumed true generation process in (1). To this end, the final learned policy π needs to be robust with respect to corruption in labels. We consider a corruption model similar to strong corruption model from robust statistics literature (Diakonikolas & Kane, 2023), which roughly says that an adversary can adaptively corrupt the labels of a fraction of samples, by inspecting the samples.
Definition 3.3 (Label Corruption Model). Let α ∈ [0, 1/2].
We consider an α-corruption model: an adversary can inspect the samples in a preference dataset of size n and then assign any label value of 0 or 1 to at most αn samples.
this section cite: ['b20', 'b13', 'b7', 'b21', 'b13', 'b41', 'b15']

Section: Interplay between Privacy and Robustness.
One key theme of this paper is to study the interplay between privacy and robustness in offline alignment. In particular, we are interested in the impact of the order between privacy protection and corruption in the labels on the suboptimality gap (cf. (2)), for both RLHF and DPO. To this end, we will mainly consider the following settings.
Definition 3.4 (CTL and LTC). Given a raw preference dataset D = (s i , a 0 i , a 1 i , y i ) n i=1 , we consider the following settings that differ in the order of privacy protection (see Definition 3.1) and corruption (see Definition 3.3). In all cases, the final input dataset for the learning algorithm will be denoted by D in = (s i , a 0 i , a 1 i , z i ) n i=1 . Corruption-then-LDP (CTL): An adversary first corrupts the labels in D to ȳi . Then, each label ȳi is privatized by a local randomizer.
LDP-then-Corruption (LTC): Each label y i in D is first privatized by a local randomizer, resulting in the private label y i . Then, the preference dataset with private labels is further corrupted by an adversary. Remark 3.5. As a last setting, one may also consider the setting where corruption happens both before and after privacy protection, which turns out to be a simple combination of the results for CTL and LTC, hence omitted in our results.
this section cite: []

Section: Reduction to Parameter Estimation
In this section, we will show that the key to establishing the suboptimality guarantees in both RLHF and DPO is a tight parameter estimation in logistic regression, under certain modeling assumptions. This allows us to focus on a singleparameter estimation problem under different settings (i.e., CTL and LTC) for both RLHF and DPO. More importantly, this unified perspective also enables us to easily see the connection and difference between RLHF and DPO.
this section cite: []

Section: Logistic Regression.
Recall that given a feature vector x i ∈ R d , under logistic regression, the label y i ∈ {0, 1} is generated according to the following probability:
P{y i = 1|x i } = σ (⟨θ true , x i ⟩) ,(3)
where σ(z) = 1 1+e -z is the sigmoid function, θ true ∈ R d is the unknown true parameter and ⟨•, •⟩ denotes the inner product of two vectors.
this section cite: []

Section: RLHF with a Linear Reward Model
We show that when the reward model in (1) is a linear function, the key to bounding the suboptimality gap in RLHF is the parameter estimation in a logistic regression problem. To start with, we formally state the linear reward model, following common definitions used in prior work (Zhu et al., 2023;Xiong et al., 2024;Cen et al., 2024;Chowdhury et al., 2023;Mandal et al., 2024).
Assumption 4.1 (Linear Reward with Boundedness). We assume that the ground truth reward r ⋆ is linear, i.e., r ⋆ (s, a) = ⟨ϕ(s, a), θ ⋆ ⟩, where ϕ(s, a) : S × A → R d is some known and fixed feature map and S, A are the state space and the action space, respectively. We also assume the following standard boundedness conditions. For all s ∈ S and a ∈ A, without loss of generality, we assume ∥ϕ(s, a)∥ ≤ 1. Moreover, we assume θ ⋆ ∈ Θ B = {θ ∈ R d : ⟨1, θ⟩ = 0, ∥θ∥ ≤ B}, where the condition ⟨1, θ⟩ = 0 is to ensure the identifiability of θ ⋆ .
Under the above assumption, we consider the standard offline RLHF algorithm, but with an additional parameter η. In particular, we consider two alternative outputs: When η = 0, the output policy is π = argmax π J(π) where J(π) = E s∼ρ,a∼π(•|s) [⟨ θ, ϕ(s, a)⟩], that is essentially a greedy algorithm with respect to an estimate θ; When η = 1, the output is π = argmax π J(π), where the objective function is defined via the principle of pessimism (Zhu et al., 2023;Jin et al., 2021;Li et al., 2024) as
J(π) = min θ∈Θ( θ,λ) E s∼ρ,a∼π(•|s) [⟨θ, ϕ(s, a)⟩] -E s∼ρ,a∼π ref (•|s) [⟨θ, ϕ(s, a)⟩],
by constructing a confidence set around an estimate θ:
Θ( θ, λ) = θ ∈ Θ B | θ -θ Σ+λI ≤ Γ(n, d, δ, λ) .
For completeness and due to space limitations, the full algorithm is given in Algorithm 2 in the Appendix B.
Here, we use a reference policy π ref because the confidence set only measures the uncertainty of the difference in reward. That is, it does not measure the uncertainty for a single stateaction pair.
We have the following key theoretical result on Algorithm 2, with its proof in Appendix E.1.
Proposition 4.2. Under Assumption 4.1, the labels {y i } i∈[n] in the preference dataset of RLHF follow the logistic regression model with θ true = θ ⋆ and
x i = ϕ(s i , a 1 i ) -ϕ(s i , a 0 i ). Algorithm 2 with η = 0 achieves SubOpt( π, π ⋆ ) ≤ 2 θ -θ true 2 ,(4)
where π ⋆ = argmax π J(π). Further, let Σ :=
1 n i x i x ⊤ i
and λ > 0 and suppose with probability at least 1 -δ the estimate θ satisfies θ -θ true Σ+λI ≤ Γ(n, d, δ, λ) .
Then, setting η = 1 in Algorithm 2, we have for any π † and ρ, with probability at least 1 -δ,
SubOpt( π, π † ) ≤ 2Γ(n, d, δ, λ) × E s∼ρ [ϕ(s, π † (s)) -ϕ(s, π ref (s))] ( Σ+λI) -1 , (6
)
for any reference policy π ref , where we define ϕ(s, π(s)) := E a∼π(•|s) [ϕ(s, a)].
We can further simplify the result in (6) by introducing the following relative condition number, which can be viewed as the natural extension of standard one (Zhang et al., 2022;Agarwal et al., 2021) to the RLHF setting.
Definition 4.3 (Relative Condition Number). For π 1 , π 2 and a feature map ϕ, we define ψ(s, a, a ′ ) = ϕ(s, a) -ϕ(s, a ′ ) and Σ π1,π2 as E s∼ρ,a∼π1(•|s),a ′ ∼π2(•|s) ψ(s, a, a ′ )ψ(s, a, a ′ ) ⊤ . (7) For any comparator policy π † and any given reference policy π ref , we define κ(π † , π ref ) := sup
w∈R d w ⊤ Σ diff π † ,π ref w w ⊤ Σ diff π sft ,π sft w .(8)
We can now simplify our previous suboptimality bound using the relative condition number above in the following corollary, with its proof given by Appendix E.2.
Corollary 4.4. Let the same assumption in Proposition 4.2 hold and further assume λ ≥ Ω d n • ln(n/δ) . For any given comparator policy π † with κ(π † , π ref ) < ∞, we can upper bound (6) as follows:
SubOpt( π, π † ) ≤ 2 √ 3 • Γ(n, d, δ, λ) • d • κ(π † , π ref ).
this section cite: ['b48', 'b43', 'b6', 'b13', 'b30', 'b48', 'b24', 'b28', 'b46', 'b0']

Section: DPO with a Log-Linear Policy Class
In this section, we will show that for a log-linear policy class (defined below), the suboptimality in DPO is also related to the parameter estimation in logistic regression.
We begin with a brief recap of DPO, following the original paper (Rafailov et al., 2023). The key idea is to reparameterize the reward model by the optimal policy of a KL-regularized problem. In particular, for the following KL-regularized optimization objective (with β > 0)
J β (π) = E s∼ρ,a∼π(•|s) r ⋆ (s, a) -β ln π(a|s) π sft (a|s) ,
the optimal solution has the closed-form expression
π ⋆ (a|s) = 1 Z β (s) π sft (a|s) exp(r ⋆ (s, a)/β),(9)
where Z β (s) = a∈A π sft (a|s) exp(r ⋆ (s, a)/β) is the normalization factor. This allows us to rewrite the reward r ⋆ in terms of π ⋆ as follows
r ⋆ (s, a) = β ln π ⋆ (a|s) π sft (a|s) + β ln Z β (s) .(10)
With the above re-parametrization of the reward using policy in (10) and BT preference model in (1), DPO (Rafailov et al., 2023) directly minimizes the following log-loss function:
L(π; π sft ) := - n i=1 1(y i = 0) ln σ β ln π(a 0 i |si) π sft (a 0 i |si) -β ln π(a 1 i |si) π sft (a 1 i |si) - n i=1 1(y i = 1) ln σ β ln π(a 1 i |si) π sft (a 1 i |si) -β ln π(a 0 i |si) π sft (a 0 i |si) .(11)
In this paper, we consider the log-linear policy class for the sake of theoretical analysis. Assumption 4.5 (Log-linear Policy Class). We assume that the optimal policy in ( 9) satisfies π ⋆ ∈ Π and π sft ∈ Π where
Π = π θ (a|s) = exp(⟨θ, ϕ(s, a)⟩) a ′ ∈A exp(⟨θ, ϕ(s, a ′ )⟩) ,(12)
is the log-linear class for some known feature map ϕ(s, a) :
S ×A → R d with ∥ϕ(s, a)∥ ≤ 1. Moreover, θ ⋆ correspond- ing to π ⋆ satisfies that θ ⋆ ∈ Θ B = {θ ∈ R d : ⟨1, θ⟩ = 0, ∥θ∥ ≤ B}, where the condition ⟨1, θ⟩ = 0 is to ensure the identifiability of θ ⋆ .
The above policy realizability assumption is equivalent to the reward model realizability. In particular, by plugging log-linear policy into (11), we can establish that the labels y i again follow from the logistic regression in (3) with proper choices of θ true and x i . In particular, we have the following formal statement, with its proof in Appendix E.3. Proposition 4.6. Under Assumption 4.5, the labels {y i } i∈[n] in the preference dataset of DPO follow the logistic regression model with θ true = β(θ ⋆ -θ sft ) with β > 0 and
x i = ϕ(s i , a 1 i ) -ϕ(s i , a 0 i ).
Suppose with probability at least 1 -δ, there exists an estimate θ that satisfies
θ -θ true Σ+λI ≤ Γ(n, d, δ, λ),(13)
where Σ :=
1 n i x i x ⊤ i and λ > 0. Then, let θ ′ = θ/β + θ sft and λ ≥ Ω d n • ln(n/δ) , the corresponding policy π = π θ ′ with probability at least 1 -δ satisfies SubOpt( π, π ⋆ ) ≤ √ 3 √ 2 • √ κ Π • B • Γ(n, d, δ, λ),
where κ Π := max π∈Π κ(π, π) is the maximum relative condition number across the entire policy class.
Remark 4.7. One can also rewrite the above bound using the maximum value of the implicit reward function, r max as
SubOpt( π, π ⋆ ) ≤ c • √ κ Π • r max β • Γ(n, d, δ, λ),
for some constant c > 0 and log-linear policy Π.
this section cite: ['b35', 'b35']

Section: Remark 4.8 (single-policy vs. all-policy concentrability).
One nice thing about the above reduction is that it allows us to easily see the key difference between RLHF and DPO. In particular, from Corollary 4.4 and Proposition 4.6, we can see that the key (and only) difference lies in the choice of relative condition number (especially when considering the typical scaling of B = O( √ d) for the parameter), which is also closely related to the "concentratability coefficient" in offline RL (Munos, 2007;Jin et al., 2021). In particular, due to the use of pessimism in offline RLHF, one can achieve a bound in terms of κ(π † , π ref ), which is related to the "single-policy concentratability" (Rashidinejad et al., 2021;Jin et al., 2021) for any comparator policy π † . On the other hand, due to the lack of uncertainty characterization in DPO, one needs "all-policy concentratability" (Chen & Jiang, 2019) κ Π in the upper bound, which is often much larger. In fact, this kind of dependence in standard DPO is shown to be necessary (Song et al., 2024).
this section cite: ['b31', 'b24', 'b36', 'b24', 'b8', 'b37']

Section: Parameter Estimation Under Private and Corrupted Labels
As motivated by the last section, we now turn to designing algorithms for providing label privacy while accurately estimating the unknown parameter θ true in logistic regression, even under corrupted labels. As we will see, the key to the design is a new loss function, which allows us to adaptively handle the privacy-robustness interplays in a unified way.
To facilitate the upcoming discussion, we formally state the general problem setup for logistic regression under private and corrupted labels.
Definition 5.1 (Private and robust parameter estimation problem). Let D be a dataset of i.i.d samples {x i , y i } n i=1 where x i ∼ µ and y i follows from the logistic regression model in (3). The input dataset
D in = {x i , z i } n i=1
is the private and corrupted version of D, following Definition 3.4. The goal here is to design a local randomizer R for privatizing labels (cf. Definition 3.1) as well as an analyzer A that receives D in outputs an estimate θ that is close to the underlying true parameter θ true , measured by a proper choice of norm. We assume the following boundedness conditions: for any i ∈ [n], ∥x i ∥ ≤ 1 and
θ true ∈ Θ B ′ = {θ ∈ R d : ⟨1, θ⟩ = 0, ∥θ∥ ≤ B ′ }.
Remark 5.2. The boundedness assumption essentially follows from the reduction in the last section. Here, we assume ∥x i ∥ ≤ 1 rather than upper bounded by 2 for simplicity and B ′ can be properly chosen for RLHF and DPO, respectively.
this section cite: []

Section: Our Algorithm
As mentioned, our choice of local randomizer R for privacy protection is the simple Random Response (RR) mechanism with parameter ε > 0 (Warner, 1965). That is, the binary output from RR equals the input with probability σ(ε) = Algorithm 1 Private and Robust Estimation 1: Procedure: ε-local label DP mechanism R 2: //Input:
U i ∈ {0, 1}, parameter: ε 3: Random response: U i = U i w.p. e ε e ε +1 1 -U i w.p. 1 e ε +1 4: Return U i 5: Procedure: Analyzer A 6: //Input: {(x i , z i )} n i=1 , parameter: ε 7: Let c(ε) = 1 2σ(ε)-1 = e ε +1 e ε -1 8: Compute θ = argmin θ∈Θ B ′ (θ) -1 n n i=1 ℓ i (θ) where ℓ i (θ) = ln(1 -σ(θ ⊤ x i )) + (z i + σ(ε) -1)c(ε)θ ⊤ x i 9: Return θ e ε
1+e ε ; otherwise, the privatized binary output differs from the input. RR satisfies the ε-local label DP guarantee (cf. Definition 3.1) (Dwork & Roth, 2014).
We now turn to the design of the analyzer A, which is responsible for outputting an estimate θ. We first point out that in the non-private non-corrupted case, the standard maximum likelihood estimator (MLE) that minimizes the loss function L(θ) = -1 n n i=1 ℓ i (θ) enjoys a good concentration (Zhu et al., 2023) with respect to θ true , where ℓ i (θ) is the standard log-loss:
ℓ i (θ) = y i log(σ(θ ⊤ x i )) + (1 -y i ) log(1 -σ(θ ⊤ x i )) = log(1 -σ(θ ⊤ x i )) + y i θ ⊤ x i .
However, due to the private labels, our analyzer is designed to minimize a new loss L(θ) = -1 n n i=1 ℓ i (θ) where
ℓ i (θ) = ln(1 -σ(θ ⊤ x i )) + (z i + σ(ε) -1)c(ε)θ ⊤ x i , (14) and c(ε) := 1 2σ(ε)-1 = e ε +1 e ε -1 .
The key difference lies in the "shifting and scaling" of the received labels z i , which, in fact, enjoys exactly the same "shifting and scaling" intuition as in mean estimation under RR, i.e., it is an unbiased estimate. Putting the above choices of R and A together, yields the final Algorithm 1 above. Remark 5.3. We remark that a similar loss (up to some scaling) has been considered in Chowdhury et al. (2023;2024). However, they are motivated from a different perspective (e.g., logit) rather than our connection to standard mean estimation under RR for local privacy (i.e., shifting and scaling). The form we use here in (14) has not appeared before. This new form not only makes it easy to see that our new loss is an unbiased estimate of the standard log loss, but also allows us to easily show that our single algorithm is adaptive to different privacy-corruption settings, i.e., it does not know the specific setting in advance.
this section cite: ['b41', 'b18', 'b48', 'b13']

Section: Estimation Error Bounds
In this section, we will establish the estimation error bounds achieved by Algorithm 1. Throughout this section, we will let θ CTL , θ LTC be the estimates outputted by Algorithm 1 under CTL and LTC respectively. Our first result is the following theorem, which characterizes the estimator error in terms of a weighted norm, with proof in Appendix E.4. Theorem 5.4. Consider the problem in Definition 5.1. For any ε > 0, α, ∈ [0, 1/2), δ ∈ (0, 1), and λ > 0, with probability at least 1 -δ, the output of Algorithm 1 achieves
θ CTL -θ true Σ+λI ≤ Γ CTL (n, d, δ, λ) := C √ α γ + c(ε) γ d + ln(1/δ) n + B ′ √ λ , θ LTC -θ true Σ+λI ≤ Γ LTC (n, d, δ, λ) := C c(ε) √ α γ + c(ε) γ d + ln(1/δ) n + B ′ √ λ , where Σ = 1 n n i=1 x i x ⊤ i , c(ε) = e ε +1 e ε -1 , γ = 1/(2 + exp(-B ′ ) + exp(B ′ )),
and C is a universal constant. Remark 5.5. First, when there is no corruption, our result matches the one in previous work on private parameter estimation (Chowdhury et al., 2023). Second, when corruption exists, the order of corruption and local privacy matters. In particular, LTC has an additional cost c(ε) in the first corruption term compared to CTL, highlighting the interplay between privacy and robustness.
Our second result is a concentration result under L 2 -norm with the additional condition of uniform coverage, which has been leveraged in prior work as well (Mandal et al., 2024;Zhang et al., 2022;Chowdhury et al., 2023). Assumption 5.6 (Uniform Coverage). There exists a positive constant ξ > 0 such that the minimum eigenvalue λ min (Σ) ≥ ξ, where
Σ := E x∼µ [xx ⊤ ].
Under the above assumption, we can have another estimation error bound for the underlying parameter, which is now in terms of L 2 -norm, with proof in Appendix E.5. Theorem 5.7. Under Assumption 5.6, for any ε > 0, α ∈ [0, 1/2), δ ∈ (0, 1), and n ≥ 8 ln(d/δ) ξ , with probability at least 1 -δ, Algorithm 1 under CTL and LTC achieves
θ CTL -θ true 2 ≤ C   α γξ + c(ε) γξ ln 1 δ n   , θ LTC -θ true 2 ≤ C   c(ε)α γξ + c(ε) γξ ln 1 δ n   .
Here, we see that the separation between CTL and LTC still exists, with an additional factor of c(ε) in LTC, illustrating a negative impact of LDP on robustness.
this section cite: ['b13', 'b30', 'b46', 'b13']

Section: Putting It All Together: Suboptimality under RLHF and DPO
In this section, we are ready to present our main results on the suboptimality gap under RLHF and DPO by combining our reduction results with estimation error bounds.
6.1. Private and Robust RLHF Theorem 6.1. Under the conditions of Corollary 4.4 and Theorem 5.4, RLHF (Algorithm 2) achieves the following suboptimality with probability at least 1 -δ
SubOpt CTL ( π, π † ) ≤ C d • κ(π † , π ref ) ×   √ α γ + c(ε) γ d + ln 1 δ n + B √ λ   , SubOpt LTC ( π, π † ) ≤ C d • κ(π † , π ref ) ×   c(ε) √ α γ + c(ε) γ d + ln 1 δ n + B √ λ   ,
for any comparator policy π † and λ ≥ Ω d n • ln(n/δ) .
The proof follows directly from the reduction result in Corollary 4.4 and estimation error bound in Theorem 5.4. To the best of our knowledge, this is the first result on the suboptimality performance of RLHF under both privacy and corruption. In particular, let λ = Θ(d/(B 2 γ 2 n)) ≥ Ω(d/n), the sample complexity part in the bounds (i.e., the last two terms) approaches zero with a rate of O( d/n), but with a multiplicative factor of c(ε) that captures the cost of privacy. Meanwhile, due to strong corruption, a non-vanishing bias term exists in all three cases in terms of corruption parameters, which illustrates an interesting interplay between privacy and robustness, discussed below.
this section cite: []

Section: Separation between CTL and LTC.
One key observation is that LDP before corruption leads to an additional c(ε) factor in the bias term, which mimics the same phenomena in private and robust mean estimation problems (Zhou & Zhang, 2024;Cheu et al., 2021).
Comparisons with Prior Work. We now highlight our contributions even in robust-only or private-only RLHF, by comparing our result above with existing ones where privacy and robustness are separately considered.
1. Robust RLHF: To our best knowledge, only recent work (Mandal et al., 2024) establishes theoretical suboptimality bounds for RLHF under adversarial corruption. In particular, it takes a linear MDP view (rather than our linear bandit view) of RLHF under strong corruption of both features and labels. Under the same relative condition number assumption, their dependence on α is O(α 1/4 ) when reduced from MDP to bandit. In contrast, our result gives a better dependence O( √ α), although only with label corruption. It is worth noting that this O( √ α) dependence is state-of-the-art even in the easier setting of standard offline reinforcement learning (Zhang et al., 2022). Moreover, our Algorithm 1 is much simpler than the one in (Mandal et al., 2024). Thus, a fair conclusion here could be that our result offers a better algorithm and theoretical result in the easier label-only corruption setting.
this section cite: ['b47', 'b11', 'b30', 'b46', 'b30']

Section: Private RLHF:
To our best knowledge, we are unaware of prior work that explicitly states the private suboptimality of RLHF in terms of relative condition number, often used in the standard offline RL. The most related one is Chowdhury et al. (2023), which generalizes the non-private RLHF in Zhu et al. (2023) to the same locally private one as ours. However, both Chowdhury et al. (2023) and Zhu et al. (2023) state their suboptimality as
SubOpt( π, π ⋆ ) ≤ ∥E s∼ρ [ϕ(s, π ⋆ (s)) -v]∥ ( Σ+λI) -1 × 2F (n, d, δ, λ),(15)
for any chosen reference vector v ∈ R d and some function F . This is similar to our intermediate result in (6) but has some key differences. One potential issue in (15) is that it does not offer clear guidance on choosing the important vector v. In particular, if v = 0, then the suboptimality may not converge to zero as n → ∞. This is because in both papers, λ has to be on the order of 1/n so as to ensure that F (n, d, δ, λ) ≤ O(1/ √ n). However, in this case, if the minimum eigenvalue of the empirical matrix Σ is small, the norm term ∥E s∼ρ [ϕ(s, π ⋆ (s)) -v]∥ ( Σ+λI) -1 can be on the order of √ n, given the choice of λ. To partially address this, Zhu et al. (2023) suggest a heuristic way of selecting v as the most common feature vector that appears in the data set. In contrast, we consider a reference policy π ref and offer a theory-grounded rule for selecting it via relative condition number along with Corollary 4.4. Our next result is the suboptimality in RLHF under the assumption of uniform coverage (cf. Assumption 5.6). Theorem 6.2. Under the conditions of Proposition 4.2 and for n ≥ 8 ln(d/δ) ξ , RLHF (Algorithm 2) achieves the following suboptimality with probability at least 1 -δ
SubOpt CTL ( π, π ⋆ ) ≤ C   α γξ + c(ε) γξ ln 1 δ n   , SubOpt LTC ( π, π ⋆ ) ≤ C   c(ε)α γξ + c(ε) γξ ln 1 δ n   .
The proof follows directly from Proposition 4.2 and Theorem 5.7. Compared with Theorem 6.1, the corruption term becomes α (with a factor of 1/ξ) rather than √ α while the concentration part has no explicit dependence on d but with 1/ξ factor, which however implicitly depends on d. As before, a separation exists between CTL and LTC, due to the additional c(ε) factor in LTC. It is worth noting that the O(α/ξ) dependence matches the best existing result in standard offline RL under corruption in Zhang et al. (2022).
Comparisons with Prior Work. Mandal et al. (2024) also consider the uniform coverage case and establish a bias corruption term on the order of
√ dα 1-o(1) ξ
when reduced from their MDP to bandit setting. In contrast, in our labelcorruption setting, we have no explicit dependence on d and a better dependence on α. Moreover, we highlight that the missing dependence of 1/γ in Mandal et al. (2024) is actually due to an error in their proof (see Appendix G for a detailed discussion). That is, the correct bound of their algorithm also has a 1/γ factor. In the context of private RLHF under uniform coverage, our bound matches the stateof-the-art in Chowdhury et al. (2023) when the corruption parameter is zero.
this section cite: ['b48', 'b13', 'b48', 'b48', 'b46', 'b30', 'b30', 'b13']

Section: Private and Robust DPO
Thanks to our reduction result, we can also leverage the estimation error bound to give the first result on suboptimality in DPO-style algorithms under privacy and corruption.
Theorem 6.3. Under the conditions of Proposition 4.6, the policy corresponding to the output of Algorithm 1 achieves the following suboptimality with probability at least 1 -δ The proof follows from Proposition 4.6 and Theorem 5.4 with B ′ = O(βB). To our knowledge, this is the first theoretical result on DPO-style algorithms under privacy and corruption. As before, we can see that the interplay of local privacy and adversarial corruption introduces a separation between CTL and LTC by a factor of c(ε). Moreover, our result also significantly advances the state-of-the-art for DPOstyle algorithms under privacy or corruption separately, as discussed in detail below.
SubOpt CTL ( π, π ⋆ ) ≤ C • B √ κ Π ×   √ α γ + c(ε) γ d + ln 1 δ n + βB √ λ   , SubOpt LTC ( π, π ⋆ ) ≤ C • B √ κ Π ×   c(ε) √ α γ + c(ε) γ d + ln 1 δ n + βB √ λ   , for β > 0, λ ≥ Ω d n • ln(n/δ) , γ = 1/(2 + exp(-βB) + exp(βB)),
Private DPO. Consider α = 0, λ = Θ(d/(β 2 B 2 γ 2 n)) ≥ Ω(d/n),
we obtain the first suboptimality for private DPO with rate O(1/γ • c(ε) d/n • √ κ Π ), where c(ε) is the additional cost due to local privacy. The rate matches the best possible non-private one as ε → ∞ (Song et al., 2024). Robust DPO. To the best of our knowledge, only the recent work by Chowdhury et al. (2024) provides a formal theoretical bound on the suboptimality of rDPO under label corruption. In particular, it considers the random-flipping corruption model (i.e., with some known probability, the true label is flipped). This is a much weaker model than ours and, in fact, is equivalent to local privacy after re-parameterization. Under this weaker model, Chowdhury et al. (2024) only established a suboptimal rate of O(1/n 1/4 ) in the general case, while our result implies a rate of O(1/n 1/2 ) (by using our private DPO result above) under the same corruption model. Moreover, moving from this weaker corruption model to a corruption model in the robust statistics literature (i.e., strong corruption model), our result above shows that rDPO now suffers a non-vanishing bias term. Practical Implementation and Experiments. Given that Theorem 6.3 establishes the SOTA theoretical results of rDPO in both private and corruption cases, under the loglinear policy. One may also interested in its empirical performance in general with neural nets as the policy class. We have a series of experiments (see Appendix D for details), which demonstrate some interesting results.
this section cite: []

Section: Discussion and Conclusion
While we present only upper bound results in the main body, we briefly discuss their tightness here; for further details, please refer to Appendix C. First, when α = 0, the additional factor c(ε) due to privacy matches the minimax lower bound established in Chowdhury et al. (2023). Furthermore, the dependence on 1/γ = Θ(e B ) = Θ(e rmax ) appears in nearly all existing results on both offline and online RLHF (Zhu et al., 2023;Zhan et al., 2023;Xie et al., 2024;Pacchiano et al., 2021;Chen et al., 2022), stemming from the non-linearity of the Bradley-Terry model. Second, in the limit ε → ∞ (non-private case), our dependence on α is O( √ α) and O(α/ζ) (under uniform coverage), both of which align with state-of-the-art results in standard offline RL settings, where rewards rather than preferences are observed. In fact, we conjecture that the O(α/ζ) dependence is optimal. Third, regarding the separation between CTL and LTC, the conclusion is nuanced. We tend to believe that the additional factor c(ε) in the uniform coverage case is tight, as it matches the known result in mean estimation and offline bandits (Zhou & Zhang, 2024). However, under the O( √ α) dependence without coverage, we hypothesize that achieving an O( c(ε)) separation-rather than O(c(ε))is possible, presenting an exciting direction for future work. Looking ahead, our reduction analysis and new results on private and robust alignment may serve as key benchmarks and inspire further research in this domain.
this section cite: ['b13', 'b48', 'b45', 'b42', 'b33', 'b10', 'b47']

Section: References
Ref_id:b0 Title: On the theory of policy gradient methods: Optimality, approximation, and distribution shift Year: (2021)
Ref_id:b1 Title: Trimmed maximum likelihood estimation for robust learning in generalized linear models Year: (2022)
Ref_id:b2 Title: Training a helpful and harmless assistant with reinforcement learning from human feedback Year: (2022)
Ref_id:b3 Title: Rank analysis of incomplete block designs: I. the method of paired comparisons Year: (1952)
Ref_id:b4 Title: Robust reinforcement learning from corrupted human feedback Year: (2024)
Ref_id:b5 Title: Open problems and fundamental limitations of reinforcement learning from human feedback Year: (2023)
Ref_id:b6 Title: Value-incentivized preference optimization: A unified approach to online and offline RLHF Year: (2024)
Ref_id:b7 Title: Sample complexity bounds for differentially private learning Year: (2011)
Ref_id:b8 Title: Information-theoretic considerations in batch reinforcement learning Year: (2019)
Ref_id:b9 Title: Classification under misspecification: Halfspaces, generalized linear models, and connections to evolvability Year: (2020)
Ref_id:b10 Title: Human-in-the-loop: Provably efficient preference-based reinforcement learning with general function approximation Year: (2022)
Ref_id:b11 Title: Manipulation attacks in local differential privacy Year: (2021)
Ref_id:b12 Title: Robust estimation of discrete distributions under local differential privacy Year: (2023)
Ref_id:b13 Title: Differentially private reward estimation with preference feedback Year: (2023)
Ref_id:b14 Title: Provably robust DPO: Aligning language models with noisy feedback Year: (2024)
Ref_id:b15 Title: Algorithmic highdimensional robust statistics Year: (2023)
Ref_id:b16 Title: Local privacy and statistical minimax rates Year: (2013)
Ref_id:b17 Title: Minimax optimal procedures for locally private estimation Year: (2018)
Ref_id:b18 Title: The algorithmic foundations of differential privacy Year: (2014)
Ref_id:b19 Title: Robust logistic regression and classification Year: (2014)
Ref_id:b20 Title: Exposing privacy gaps: Membership inference attack on preference data for LLM alignment Year: (2024)
Ref_id:b21 Title: Deep learning with label differential privacy Year: (2021)
Ref_id:b22 Title: A tail inequality for quadratic forms of subgaussian random vectors Year: (2011)
Ref_id:b23 Title: Correcting the mythos of KL-regularization: Direct alignment without overparameterization via Chi-squared preference optimization Year: (2024)
Ref_id:b24 Title: Is pessimism provably efficient for offline RL Year: (2021)
Ref_id:b25 Title: Extremal mechanisms for local differential privacy Year: (2014)
Ref_id:b26 Title: What can we learn privately? Year: (2011)
Ref_id:b27 Title: The history and risks of reinforcement learning and human feedback Year: (2023)
Ref_id:b28 Title: Settling the sample complexity of model-based offline reinforcement learning Year: (2024)
Ref_id:b29 Title: On robustness and local differential privacy Year: (2023)
Ref_id:b30 Title: Corruption robust offline reinforcement learning with human feedback Year: (2024)
Ref_id:b31 Title: Performance bounds in L p -norm for approximate value iteration Year: (2007)
Ref_id:b32 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b33 Title: Dueling RL: reinforcement learning with trajectory preferences Year: (2021)
Ref_id:b34 Title: Robust estimation via robust gradient estimation Year: (2020)
Ref_id:b35 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b36 Title: Bridging offline reinforcement learning and imitation learning: A tale of pessimism Year: (2021)
Ref_id:b37 Title: The importance of online data: Understanding preference fine-tuning via coverage Year: (2024)
Ref_id:b38 Title: An introduction to matrix concentration inequalities Year: (2015)
Ref_id:b39 Title: Transformer Reinforcement Learning Year: (2020)
Ref_id:b40 Title: A comprehensive survey of LLM alignment techniques: RLHF, RLAIF, PPO, DPO and more Year: (2024)
Ref_id:b41 Title: Randomized response: A survey technique for eliminating evasive answer bias Year: (1965)
Ref_id:b42 Title: Exploratory preference optimization: Harnessing implicit Q*-approximation for sampleefficient RLHF Year: (2024)
Ref_id:b43 Title: Iterative preference learning from human feedback: Bridging theory and practice for RLHF under KL-constraint Year: (2024)
Ref_id:b44 Title: Cautiously optimistic policy optimization and exploration with linear function approximation Year: (2021)
Ref_id:b45 Title: Provable offline preference-based reinforcement learning Year: (2023)
Ref_id:b46 Title: Robust policy gradient against strong data corruption Year: (2021)
Ref_id:b47 Title: Locally private and robust multiarmed bandits Year: (2024)
Ref_id:b48 Title: Principled reinforcement learning with human feedback from pairwise or K-wise comparisons Year: (2023)
Ref_id:b49 Title: Category: Education & Skill Development Prompt: "You're saving $5,000 to attend a data visualization course. How do you proceed?" Chosen Year: (2019)
