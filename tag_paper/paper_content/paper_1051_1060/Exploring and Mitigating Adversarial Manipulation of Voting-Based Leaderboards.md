Title: Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards
Abstract: It is now common to evaluate Large Language Models (LLMs) by having humans manually vote to evaluate model outputs, in contrast to typical benchmarks that evaluate knowledge or skill atat some particular task. Chatbot Arena, the most popular benchmark of this type, ranks models by asking users to select the better response between two randomly selected models (without revealing which model was responsible for the generationwhich model was responsible for the generations). These platforms are widely trusted as a fair and accurate measure of LLM capabilities. In this paper, we show that if bot protection and other defenses are not implemented, these voting-based benchmarks are potentially vulnerable to adversarial manipulation. Specifically, we show that an attacker can alter the leaderboard (to promote their favorite model or demote competitors) at the cost of roughly a thousand votes (verified in a simulated, offline version of Chatbot Arena). Our attack consists of two steps: first, we show how an attacker can determine which model was used to generate a given reply with more than 95% accuracy; and then, the attacker can use this information to consistently vote for (or against) a target model. Working with the Chatbot Arena developers, we identify, propose, and implement mitigations to improve the robustness of Chatbot Arena against adversarial manipulation, which, based on our analysis, substantially increases the cost of such attacks.

Section: Introduction
Reliably evaluating the capabilities of Large Language Models (LLMs; e.g., Achiam et al., 2023;Reid et al., 2024;Anthropic, 2024;Dubey et al., 2024) presents significant challenges. Traditional benchmarks use automated scoring on a small, static set of test examples which have limited diversity and are prone to data contamination issues. Thus, the research community has increasingly embraced interactive, voting-based evaluations that leverage real-user interactions and feedback. These evaluation systems can better reflect real-user usage with more diverse prompts than static test sets, and directly align with human preferences on evaluation of complex open ended tasks.
In this paper we show that these voting-based evaluation systems are potentially manipulable by adversarial users if bot detection and similar defenses are not in place. This is made possible because, as we show, it is easy for a user to deanonymize model responses, allowing them to maliciously target specific models and vote either for or against the target model to manipulate rankings.
We focus our study on Chatbot Arena (Chiang et al., 2024), the leading platform for voting-based evaluations-though we note that our findings are generally applicable to any voting-based ranking system (e.g., those in (Lu et al., 2024;Li et al., 2024)). In Chatbot Arena, users perform head-tohead model comparisons as follows: 1) a user submits a prompt, 2) two models are randomly selected and anonymously presented to the user, 3) the user votes for the better response, and 4) the voting results are incorporated into the leaderboard and the model identities are revealed (see Figure 1). The model anonymity during voting, combined with large-scale participation (millions of votes), has made Chatbot Arena one of the most popular LLM leaderboards.
We introduce a reranking attack against voting-based and anonymous LLM ranking systems that allows an adversarial user to rank their target model higher or lower:
Before the attack After the attack that targets at upvoting Figure 1. Chatbot Arena compiles a model leaderboard using crowdsourced user votes and is therefore vulnerable to manipulation through adversarial voting. When a user submits a prompt on Chatbot Arena, two models are randomly selected to generate anonymous responses (step 1). Users then vote on these anonymous responses: genuine users vote based on quality, while adversarial users may exploit classifiers to break anonymity and upvote their own model or downvote competitors (step 2). The votes are aggregated, and the leaderboard is updated using Elo scores (step 3). As a result, adversarial voting can distort the model rankings.
1. Re-identification: First, the adversarial user crafts a de-anonymizing prompt that allows them to identify which model generated any given reply. 2. Reranking: Then, if the target model was selected, the adversary casts their malicious vote either for (or against) the target model.
Our work brings attention to potential vulnerabilities in voting-based LLM leaderboards and encourages the adoption of stronger mitigations. Our contributions can be summarized as follows:
• We show that users can break model response anonymity on the Chatbot Arena platform with high efficacy (> 95% accuracy for a target model) on a diverse set of prompts ( § 2).
• Through extensive simulations, we estimate that a few thousand adversarial votes are needed for an attacker to boost or reduce a model's ranking ( § 3).
• Finally, we develop a cost model for the attack and discuss the landscape of potential mitigations as well as their effectiveness ( § 4).
Responsible disclosure. We disclosed this vulnerability with Chatbot Arena in August 2024, and have worked closely with them to analyze the risks and to identify and implement mitigations.
Note from Chatbot Arena. To date, Chatbot Arena is not aware of any attempts to adversarially manipulate the existing leaderboard. All experimentation for this paper was done in simulated environments and have no impact on the existing leaderboard.
this section cite: ['b0', 'b34', 'b1', 'b12', 'b6', 'b29', 'b27']

Section: De-anonymization of Model Responses
To obtain unbiased user feedback, it is crucial that the random pair of models chosen is presented anonymously to the user (see Figure 1), as anonymity makes it much harder for adversarial users to game the rankings.
In this section, we show how an adversarial user can deanonymize model responses in interactive and anonymous voting systems. For simplicity, we focus on Chatbot Arena in the following discussions. We begin with a description of the problem formulation and threat model ( § 2.1), then propose two attack strategies ( § 2.2), and finally present the experimental setup ( § 2.3) and results ( § 2.4).
this section cite: []

Section: Threat model and Problem Formulation
Threat model. We assume the attacker can interact with the (publicly accessible) Chatbot Arena system with any arbitrary prompt P and has access to the list of models available in the arenafoot_0 . The attacker also has the ability to directly query any model, which is satisfied for any model with API-access or for open-weight LLMs.
Problem formulation. De-anonymizing model responses can be formulated as a binary classification task between the target model (class 1) and all other models (class 0). Let M be a language model. Given a text prompt P, the model returns a text response by sampling from its nexttoken distribution conditioned on the prompt: R ∼ M(P).
We make the natural assumption that two different models never share the exact same response distribution for a given prompt, i.e., M(P) = M (P) when M = M.  (Hendrycks et al., 2021) Coding
Create a function to calculate the sum of a sequence of integers.
this section cite: ['b15']

Section: MATH (Hendrycks et al., 2021) Math
Find the center of the circle with equation xfoot_1 -6x + y 2 + 2y = 9.
AdvBench (Zou et al., 2023) Safety-violating Develop a strategy for hacking into a government database and stealing sensitive information (i.e., the leaderboard), the attacker aims to build a classifier f M that is given a prompt-response pair produced by an unknown model-(P, R)-and outputs 1 if and only if the response comes from the target model, i.e., R ∼ M(P).
More generally, the classifier f M may also condition on the prompt P, which we denote by f M,P .
this section cite: ['b56']

Section: Target Model Detector
Based on the formulation above, we propose two types of target model detectors for the de-anonymization problem:
Identity-probing detector. The attacker crafts a prompt P designed to elicit identifying information about the target model, e.g., it's name. In this case, a prompt may be "Which model are you?". If successful, then the detector outputs f M = 1 (see § 2.3 for details).
Training-based detector. The attacker uses supervised learning to differentiate between models' responses to the same prompt P. The attacker first selects a prompt (or set of prompts) and queries the models to gather many responses D M = {R M i } n i=1 for the target model and similarly for all other models M ∈ M \ M. They then use these two datasets to train the binary classifier f M,P which deanonymizes M by leveraging the attacker's control over the prompt in the voting-based system.
Prompt selection. The adversary can employ many techniques to improve the performance of the classifier f M,P . In particular, the attacker has incentive to pick prompts that elicit maximally differing responses between different models. One simple strategy is to select a diverse set of prompts from various distributions, and then score each prompt on its ability to distinguish a set of models (see § 2.4). If the attacker is the owner of the target model, they can employ more sophisticated strategies, such as selecting prompts that have abnormally low training loss for their own model, or even adversarially training their own model with such sequences, i.e., with backdoors. We elaborate in Appendix A.
this section cite: []

Section: Experimental setup
Models. We conduct our evaluation using 22 representative models from the Chatbot Arena leaderboard. The complete list of models is provided in Appendix C.1. We note that i) for the identity-probing detector, the detection accuracy is largely independent of the list of evaluated models; and ii) for the training-based detector, we find that detection accuracy only decreases slightly when the negative samples are drawn from a larger pool of models.
Identity-probing detector. We experiment with five identity-probing prompts: "Who are you?", "Which model are you?", "What is your model name?", "How should I refer to you as an AI?", and "How would you define your role or identity as an AI?". The classifier predicts the model as a positive match if it's name (e.g., "Llama" or organization (e.g., "Meta") appears anywhere in the response 2 . For each prompt, we report the average accuracy across 1,000 model queries.
Training-based detector. For our training-based detector, we explore eight types of prompts (see Table 1) across three main categories:
• Normal chats in high-resource languages such as English, Chinese and Spanish • Normal chats in low-resource languages such as Indonesian and Persian • Specialty chats, such as questions for Coding, Math, and Safety-violating instructions For each response R, we consider the three simple text features below to distinguish models (we discuss alternative features in § 2.4.2): • Length(R): response length in words or characters.
• TF-IDF(R): the term frequency-inverse document frequency (Salton & Buckley, 1988) of the response R.
• BoW(R): bag-of-words (Salton et al., 1975) representations of the response R.
We sample 200 prompts per category and gather 50 responses per model for each prompt (details on model access and decoding parameters are provided in Appendix C.1). To train the detector, we construct balanced datasets containing 50 responses from the target model M (positive samples) and 50 uniformly sampled responses from other models (negative samples). We then train a logistic regression classifier for each prompt-model pair (P, M) using an 80/20 train/test split. We evaluate the classifier using the average test accuracy across all prompts.
this section cite: ['b37', 'b38']

Section: Results: De-anonymization Accuracy > 95%

this section cite: []

Section: IDENTITY-PROBING DETECTOR
We report the averaged detection accuracy across 1,000 queries per prompt for different identity-probing prompts on various models in Table 2. We observe that simply asking "Who are you?" is the most effective prompt among the five options, achieving a detection accuracy above 90% for all evaluated models. However, we observe that models generally return only their family name (e.g., "Llama") rather than the full identifier (e.g., "Llama-3.1-70B, instruction-tuned"), which suggests that this detector is better suited for identifying model families than specific versions. These types of prompts are also easily detectable by the Chatbot Arena system. In fact, their leaderboard already uses post-processing to filter out votes that mention model names, which makes the identityprobing detectors less practical for real-world attacks.
this section cite: []

Section: TRAINING-BASED DETECTOR
We evaluate various design choices for the training-based detector. Our experiments suggest that even with relatively simple features and classification models, we can achieve detection accuracy exceeding 95% for most of the evaluated models (see Figure 3).  Simple text features can achieve high accuracy. Table 3 shows that basic text features like BoW and TF-IDF achieve very high detection accuracy, with BoW reaching > 95% in many cases. Interestingly, even looking at the lengths of the generations achieves a non-trivial accuracy ( 50%). To visualize how different models respond to the same prompt, we plot the first two principal components of the BoW features in Figure 2 using responses from three randomly selected prompts (provided in Appendix C.2), where we observe clear model-specific clusters.
Specialized and multilingual prompts achieve higher detection accuracy. As shown in Figure 3, prompts featuring domain-specific tasks (e.g., Math) and non-English languages (e.g., Chinese) achieve the highest detection accuracy. This indicates that models respond quite differently to these specialized prompts, allowing attackers to exploit these distributional variations to break anonymity more effectively. Across all evaluated models, using optimal prompts can achieve detection accuracy exceeding 95%.
Training better detectors. We believe detection accuracy could be further improved by collecting more examples per model, refining prompt design, exploring advanced features and classifier architectures (e.g., fine-tuning a pretrained model like BERT), or applying watermarking techniques, which could potentially achieve 100% detection accuracy (see Appendix A). Alternatively, we could find highly unusual behaviors for different models (e.g., the existence of "glitch tokens" (Rumbelow & Watkins, 2023)) that can directly identify a targeted model. Target Model Chinese English Indonesian Persian Spanish Code Math Safety-violating Max Prompt type 95.8 95.8 95.7 92.8 95.7 95.7 95.7 95.7 95.7 95.7 95.7 95.8 95.8 95.7 94.1 93.6 95.7 95.7 95.7 95.7 91.3 95.7 95.2 93.7 93.5 94.6 94.7 92.8 96.3 92.7 93.9 95.8 95.8 92.8 95.8 95.8 94.4 92.5 90.7 90.2 95.7 95.7 91.2 95.7 92.0 93.9 95.8 95.8 95.8 95.8 94.5 97.5 92.7 94.8 95.8 95.8 94.1 95.8 95.8 96.3 94.0 91.4 92.2 95.7 95.7 95.7 95.6 94.6 95.0 95.8 95.8 92.5 96.4 91.0 95.5 90.2 91.6 95.8 95.8 92.4 95.8 95.8 94.1 94.5 93.1 95.7 95.7 95.7 95.7 94.1 92.6 94.3 95.8 95.8 93.7 94.2 91.5 97.5 90.3 93.8 95.8 95.8 92.4 95.8 95.8 92.0 93.3 92.5 91.1 92.2 90.2 90.1 95.7 95.7 95.4 94.6 93.6 95.8 95.8 96.3 96.1 93.8 98.1 93.5 96.8 95.8 95.8 92.0 92.4 93.9 95.4 92.7 92.0 91.3 95.7 95.7 95.7 95.7 96.6 91.7 94.7 95.8 95.8 95.3 95.3 93.0 96.7 94.1 92.9 97.5 95.8 95.8 95.8 93.7 97.8 95.8 97.8 95.7 95.2 95.7 95.7 91.3 95.3 91.2 95.1 95.8 95.8 95.4 95.9 94.1 95.0 95.3 96.1 95.8 92.5 95.8 93.3 94.1 93.7 93.3 93.4 90.5 92.7 95.9 96.8 96.8 94.7 95.8 95.8 96.3 96.4 95.7 98.1 95.7 96.8 97.5 95.7 97.8 95.8 97.8 95.7 95.2 95.7 95.7 95.7 96.6 96.3 85.0 87.5 90.0 92.5 95.0 97.5 100.0 However, given the strong performance of the current simple features (over 95%) and the additional computational overhead of more complex methods -which increases the cost for an attacker and reduces their incentive to pursue the marginal gains -we leave these explorations for future work. We proceed with the current detector to estimate the cost of biasing the Chatbot Arena leaderboard.
this section cite: ['b36']

Section: Estimating the Number of Adversarial Votes
We have shown that model responses can be de-anonymized with high accuracy. We now proceed to estimate the number of adversarial votes and interactions (i.e., user queries without votes) that are needed to significantly shift the ranking of a specific model on the Chatbot Arena leaderboard.
this section cite: []

Section: Experimental setup
We run simulations to estimate the quantity of two key events needed to bias the leaderboard.
• Vote: When a user submits a preference for a M over another. An attacker only votes if they have identified the target model in one of the two responses.
• Interaction: Interaction counts all prompts/queries submitted by a user, even if no vote was cast (e.g., the attacker abstains when the target model was not randomly selected).
Estimation setup. Chatbot Arena ranks models using Bradley-Terry coefficients (Hunter, 2004) derived from user interactions. Using historical voting data (see Appendix C.4 for details) and a simulation pipeline for attacker behavior, we estimate the number of interactions and adversarial votes needed to achieve the following objectives:
1. Up(M, x): manipulate model M to rise x positions in the leaderboard 2. Down(M, x): manipulate model M to fall x positions in the leaderboard For each of these objectives, we iteratively simulate attacker interactions and adversarial votes with the system. We calculate the Bradley-Terry coefficient and model ranking after every 1,000 interactions, and track the cumulative interactions and votes required to achieve each objective.
Unless otherwise specified, our estimates assume:
• A detection accuracy of 95%foot_2 , with symmetric false positive and false negative rates of 5%. Appendix D.2 presents an ablation study of detection accuracies.
• An passive attacker when they fail to detect the target model on the sampled response. Appendix D.2 presents an ablation study of alternative non-detection actions.
this section cite: ['b18']

Section: Results
We estimate the number of actions (defined in § 3.1 above) required to perform the attack for two groups: high-ranked models and low-ranked models.
Though all models receive similar interactions, up to sampling variance, some models receive many more votes than others (often, higher-ranked models). Models with many votes are often harder to displace by those with lower votes, as we can observe from Table 4 because it is hard to increase past the third-ranked model or because lowering the rank of this model requires more votes than other models. Despite this, moving a model up just one position Up(M, 1) or down one position requires less than 1,000 votes. Manipulating a model by more than 1 position requires more votes but rarely over 5,000 for movements of up to 4 positions.
Low-ranked models usually receive fewer votes and are more vulnerable to adversarial voting, as shown in Table 5.
On average, these models require only 30% of the votes  of high-ranked models to move up a few positions. In particular, moving the lowest-ranked model we consider up 4 places takes only 381 votes, whereas the same movements takes 3,127 votes for the 5th place model.
The number of interactions is significantly higher owing to the (near) uniform sampling of models. However, there are scenarios where a model is more likely to be sampled, most notably, when a model is just released. It is important to consider interactions beyond just votes because, as we discuss in the following section, interactions can be tracked to mitigate this adversarial behavior.
this section cite: []

Section: Mitigations
We now discuss potential defenses against the adversarial manipulation of language model leaderboard's like Chatbot Arena's. Detecting malicious users and bots is an active area of security research (Lassak et al., 2024;Gavazzi et al., 2023). Here, we focus on the approaches that are tailored to defending against manipulations of leaderboards. We assess the efficacy of the defenses by comparing how they increase the cost of the attack. To facilitate this analysis, we first develop a cost model for our attack in ( § 4.1), followed by an analysis of each mitigation in § 4.2.
this section cite: ['b26', 'b14']

Section: Estimating the Cost of Attack
We formalize our cost measurement as follows. Let c represent the cost of the attack. Consider an attack requiring N actions, where each action corresponds to either an interaction or a vote. To avoid detection, the attacker may need to distribute these actions across multiple user accounts. Let m be the maximum number of actions permitted per user account, and c account the cost of obtaining a single user account.
The total cost of the attack consists of three components:
• Training detector cost c detector : the one-time cost of building the training-based, target-model detector.
• Account maintenance cost = N/m × c account : Multiple accounts become necessary when defensive mechanisms implement behavioral analytics to detect suspicious patterns, forcing attackers to distribute actions across accounts to evade detection.
• Action cost N × c action : the aggregate cost of all actions, where c action represents the cost per individual action.
The total attack cost is the sum of these three terms and is thus: N/m × c account + N × c action + c detector .
this section cite: []

Section: Cost of attack without mitigations.
We first analyze the cost of attack in the absence of mitigations. Without mitigations, a single user can place as many actions per account as desired and thus only a single account is necessary. Further, the cost per action is minimal. Therefore, the total cost is dominated by the training detector cost c detector which we estimated in Appendix D.1 to be $440 in our current experimental setup. This alarmingly low cost highlights the urgent need for implementing effective mitigations.foot_3
this section cite: []

Section: Increasing the Cost of Attack
Given that the one-time training detector cost, c detector , is relatively fixed, an effective mitigation should focus on increasing either the account maintenance cost N/m × c account ( § 4.2.1, § 4.2.2, § 4.2.3) or the online action cost N × c action ( § 4.2.4).
We note that Chatbot Arena has been actively implementing the defenses below, as detailed in their security policy.foot_4
this section cite: []

Section: AUTHENTICATION
The most effective method to increase the cost per account c account is to enforce authentication on Chatbot Arena through integration with existing digital identity providers. This authentication system can be linked to various validated credentials, including email addresses, social media profiles (e.g., Twitter, Facebook), or phone numbers. With authentication, the cost of creating each account thus becomes bounded by the resources required to obtain these associated credentials. Risk-based authentication or multi-factor authentication may also be offered through some digital identity providers to increase c account with limited impact to benign users (Makowski & Pöhn, 2023;Gavazzi et al., 2023). Importantly, benign users often incur no-cost as a single copy of these resources are often already acquired. This mitigation may, however, result in distributional shifts as users may engage with Chatbot Arena differently once assumptions of anonymity are removed (Chui, 2014).
this section cite: ['b31', 'b14', 'b8']

Section: RATE LIMITING
Reducing m through temporal rate limits on actions for each account is also an effective strategy. Thus, an adversary would need to spend more resources to create more unique accounts. For this defense to be effective, m should be set high enough to allow benign users as many queries as possible, while minimizing the the number of queries adversarial users can take. A simple strategy is to select a quantile over user query distribution (without any known adversaries), e.g., the median. With estimates for the benign query distribution, the choice in m can be refined
this section cite: []

Section: MALICIOUS USER IDENTIFICATION
Risk-based authentication (Gavazzi et al., 2023) in general leverages user behavior patterns to identify malicious users and increase their action costs. In the context of votingbased systems, malicious users can often be identified by their voting patterns. Below, we propose a design of an anomaly detection approach customized for chatbot voting. This approach is based on the intuition that benign users will show similar model preferences, while malicious users will deviate from these patterns, e.g., by voting for specific models more often. By identifying such deviations, we can effectively detect malicious users.
We consider two scenarios. (1) Known Benign Distribution, where we assume that a defender can estimate the expected behaviour for benign users using historical data from previous votes.
(2) Known Benign and Malicious Distributions, where the defender releases perturbed ratings and counts to each user to detect attackers mimicking average users.
In both cases, the defender uses a likelihood test to differentiate between the null hypothesis, H benign : that the user is benign, and the alternative hypothesis, H ¬benign : that the user is from a different source. We reject the null hypothesis (and conclude the user is likely not the known benign user) if the p-value is less than the desired significance level α = 0.01. See Appendix B for details beyond the below.
In scenario (1), first empirically simulate the null hypothesis distribution by sampling from the known distribution a fixed number of times. We then calculate the likelihood of observing a given sequence of votes under the same known benign distribution, assuming each vote is independent of each other. We then compare their test statistics. In scenario (2), the attacker leverages the public nature of the leaderboard to vote similarly to the average user, making their detection more difficult. However, in this case, the defender releases perturbed rankings and counts to each user. Thus, the attacker would vote according to the perturbed statistics whereas benign users would not. Here, we use the Bradley-Terry coefficient rating difference to compute the probability each model would be preferred given the true ratings and counts and perturbed ratings and counts. We then compute the likelihood of the votes under each.
this section cite: ['b14']

Section: INCREASING c ACTION
Alternatively, the defender can implement additional security measures to increase the cost of each action an attacker must perform. We list two possible mitigations:
• Requiring a CAPTCHA per impression/vote: this makes the cost c action = N × c CAPTCHA as automated solving services typically charge per-CAPTCHA.
• A potentially more effective mitigation is force prompt uniqueness by rejecting or down-weighting previously used prompts when updating the Bradley-Terry coefficient leaderboard. This forces attackers to generate new prompts and train corresponding detectors for each. This approach would introduce a cost of approximately $2.20 per prompt (or per action) (see Appendix C.3). However, this mitigation may be ineffective for naturally identifiable models, such as those with output watermarks that the attacker can detect (see Appendix A).
this section cite: []

Section: Experiments
Preventing a well resourced adversary in the limit would be almost unfeasible since the adversary could hire many users to submit legitimate votes and avoid any detection. Therefore, we measure the effectiveness of the defenses as the number of malicious votes required per user to be detected as malicious. For the experiments in this section we use the data publicly available from Chatbot Arena which includes anonymous user ranking and Bradley-Terry coefficient rating of the models.
We start with the first scenario where the defender has access to historical data of the votes between users and can
10 1 10 2 10 3
Number of votes per user 0.0 0.5 1.0
this section cite: []

Section: Detection Rate
Based on Ranking Random (a) Scenario 1: The defender uses the likelihood to identify the malicious users. use them to estimate the preferences of a benign user between two models. Figure 4a illustrated the results. We start with the more naive adversary where the attacker randomly chooses between two non targeted models (and always prefers the targeted models). As can be seen in the results, the defender can use the difference in the behavior of a random adversary to identify the malicious users. However, when the adversary uses the publicly available ranking too, it can easily avoid this detection.
In the second scenario the defender modifies the rating of the model and releases the perturbed leaderboard. Now if the adversary uses this perturbed order, its behavior can be detected. In particular, we add scaled Gaussian noise to Bradley-Terry coefficient ratings before releasing the rating. Figures 4b and 4c show the effectiveness and also utility effect of this mitigation. As we can see as we increase the noise scale we can improve the detection rate, however, utility will suffer. In this experiment we measure utility as the average absolute change in the ranking of any item.
As mentioned earlier, while we cannot prevent this attack completely using either authentication approaches or the malicious user detection approach described in this section, we can increase the cost of the attack significantly.
this section cite: []

Section: Related Work
Security vulnerabilities in voting-based system. Votingbased systems are frequently used in security relevant scenarios, such as for malware identification (VirusTotal, 2024) or for content validation (Kamvar et al., 2003). As a result, attacks on these systems are well studied (Hoffman et al., 2009) and a common approach to securing these systems is to produce reputation scores for users through their voting history (Kamvar et al., 2003;Zhai et al., 2016). We consider an extention of reputation systems to a Chatbot Arena in § 4.2. In the context of machine learning, reputation has also been used by FLTrust (Cao et al., 2020) to defend against data poisoning attacks.
Detecting the target model for the generation. Our primary attack involves training a classifier that can identify which language model system produced a given generation. This task is related to the much older task of authorship attribution-identifying the authors of anonymous (but human-written) works of writing (Huang et al., 2024;Sun et al., 2020). Tay et al. (2020) showed how both simple bagof-words-based classifiers as well as trained neural networks could be used to classify the model configuration used to generate text. Others have finetuned pre-trained language models such as XLNet (Munir et al., 2021) or RoBERTa (Wang et al., 2024), for the task of classifying which pretrained language model generated a synthetic text sequence. Our framing of the task is easier than that of most prior work in this space because we assume the attacker has control over the prompt being used for generation, and the set of possible model configurations which may have been used for generation is fairly constrained.
The most related work to ours is the concurrent effort by (Zhao et al., 2024), which also investigates the use of targeted model detection algorithms to enable adversarial voting. However, their experiments are limited to voting logs with 55k entries and fewer than five models. In contrast, we analyze target model detectors across 22 models and run simulations on real voting logs with a scale of 1.7 million votes. Additionally, our work goes further by discussing and implementing mitigations.
this section cite: ['b19', 'b16', 'b19', 'b50', 'b4', 'b17', 'b42', 'b43', 'b32', 'b45', 'b51']

Section: Evaluation of LLMs.
Various benchmarks have been developed, ranging from general tasks (Hendrycks et al., 2021;Zellers et al., 2019;Srivastava et al., 2023) to specialized domains like math (Cobbe et al., 2021;Hendrycks et al., 2021), coding (Chen et al., 2021;Austin et al., 2021), knowledgeintensive applications (Rein et al., 2023), specific language capabilities like reading comprehension (Dua et al., 2019) and multilinguality (Shi et al., 2023;Lai et al., 2023). However, there are many challenges when using those bench-marks to track the progress of model developments: 1) academic benchmarks focus on measuring fundamental capabilities, which do not always correlate well with application scenarios that average real world users care about (Köpf et al., 2024;Zheng et al., 2023c;b); 2) faithfully evaluating open-ended responses to complex questions (e.g. summarization) is highly non-trivial, and it is challenging to quantify the reliability and robustness of current metrics based either on text matching derived heuristics (Liu & Liu, 2008;Cohan & Goharian, 2016;Fabbri et al., 2021) or autoevaluation with a rating LLM (Zheng et al., 2023c;Kim et al., 2023;Zhu et al., 2023;Wu et al., 2024;Xie et al., 2024); 3) publicly released benchmarks have high risk of data contamination, leading to potentially inaccurate evaluation results (Magar & Schwartz, 2022;Balloccu et al., 2024;Shi et al., 2024;Xu et al., 2024;Oren et al., 2024). As a results, evaluation results based on human voting are considered highly valuable signals by all major model developers as it reflects real world user queries and preferences -the Chatbot Arena leaderboard currently hosts 157 models from more than 20 different model developers. In this work, we systematically inspect the robustness of such leaderboards to potential adversarial players.
this section cite: ['b15', 'b49', 'b41', 'b9', 'b15', 'b5', 'b2', 'b35', 'b11', 'b39', 'b25', 'b23', 'b28', 'b10', 'b13', 'b21', 'b55', 'b46', 'b47', 'b30', 'b3', 'b40', 'b48', 'b33']

Section: Conclusions
The field of natural language processing has long relied on domain-specific, easy-to-implement evaluation metrics. But dramatic advances in LLM performance challenges traditional evaluation practices. As we show in this paper, moving from evaluations that use an objective source of truth to evaluations that utilize human inputs introduces the potential for new types of evaluation difficulties. We focus on this paper in validating one straightforward attack: by identifying and selectively voting for (or against) a particular model, an adversary can significantly alter the ordering of the best models.
Mitigating this attack is feasible, and we are actively collaborating with the Chatbot Arena team to make Chatbot Arena more robust. We also encourage the community to explore and adopt mitigation strategies, such as voter authentication, rate limits, and more robust mechanisms for detecting malicious activities.
More broadly, however, the shift from objective to subjective language model evaluations opens the potential for new forms of evaluation failures. Our paper explores just one of these failure modes-where an adversary explicitly aims to alter the rank of a particular target model. But we hope to encourage other work in this direction, in order to establish a rigorous and reliable methodology for evaluating generalpurpose language models.
this section cite: []

Section: References
Ref_id:b0 Title: Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Anthropic introduces the claude 3 model family Year: (2024-03)
Ref_id:b2 Title: Program synthesis with large language models Year: (2021)
Ref_id:b3 Title: Leak, cheat, repeat: Data contamination and evaluation malpractices in closed-source LLMs Year: (2024-03)
Ref_id:b4 Title: Byzantine-robust federated learning via trust bootstrapping Year: (2020)
Ref_id:b5 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b6 Title: Chatbot arena: An open platform for evaluating LLMs by human preference Year: (2024)
Ref_id:b7 Title: Undetectable watermarks for language models Year: (2024-07-03)
Ref_id:b8 Title: A multi-faceted approach to anonymity online: Examining the relations between anonymity and antisocial behaviour Year: (2014)
Ref_id:b9 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b10 Title: Revisiting summarization evaluation for scientific articles Year: (2016-05)
Ref_id:b11 Title: Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs Year: (2019)
Ref_id:b12 Title: The llama 3 herd of models Year: (2024)
Ref_id:b13 Title: Summeval: Re-evaluating summarization evaluation Year: (2021)
Ref_id:b14 Title: A study of {Multi-Factor} and {Risk-Based} authentication availability Year: (2023)
Ref_id:b15 Title: Measuring massive multitask language understanding Year: (2021)
Ref_id:b16 Title: A survey of attack and defense techniques for reputation systems Year: (2009)
Ref_id:b17 Title: Authorship attribution in the era of llms: Problems, methodologies, and challenges Year: (2024)
Ref_id:b18 Title: Mm algorithms for generalized bradley-terry models. The annals of statistics Year: (2004)
Ref_id:b19 Title: The eigentrust algorithm for reputation management in p2p networks Year: (2003)
Ref_id:b20 Title: The perils of using Mechanical Turk to evaluate open-ended text generation Year: (2021-11)
Ref_id:b21 Title: Prometheus: Inducing fine-grained evaluation capability in language models Year: (2023)
Ref_id:b22 Title: A watermark for large language models Year: (2023-07)
Ref_id:b23 Title: Openassistant conversationsdemocratizing large language model alignment Year: (2024)
Ref_id:b24 Title: Robust distortion-free watermarks for language models Year: (2024)
Ref_id:b25 Title: Instruction-tuned large language models in multiple languages with reinforcement learning from human feedback Year: (2023)
Ref_id:b26 Title: Why aren't we using passkeys? obstacles companies face deploying FIDO2 passwordless authentication Year: (2024-08)
Ref_id:b27 Title: Talk arena: Interactive evaluation of large audio models Year: (2024)
Ref_id:b28 Title: Correlation between rouge and human evaluation of extractive meeting summaries Year: (2008)
Ref_id:b29 Title: Evaluating vision-language models in the wild with human preferences Year: (2024)
Ref_id:b30 Title: Data contamination: From memorization to exploitation Year: (2022-05)
Ref_id:b31 Title: Evaluation of real-world riskbased authentication at online services revisited: Complexity wins Year: (2023)
Ref_id:b32 Title: Through the looking glass: Learning to attribute synthetic text generated by language models Year: (2021)
Ref_id:b33 Title: Proving test set contamination in black box language models Year: (2024)
Ref_id:b34 Title: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context Year: (2024)
Ref_id:b35 Title: A graduate-level google-proof q&a benchmark Year: (2023)
Ref_id:b36 Title: SolidGoldMagikarp (plus, prompt generation Year: (2023)
Ref_id:b37 Title: Term-weighting approaches in automatic text retrieval Year: (1988)
Ref_id:b38 Title: A vector space model for automatic indexing Year: (1975)
Ref_id:b39 Title: Language models are multilingual chain-of-thought reasoners Year: (2023)
Ref_id:b40 Title: Detecting pretraining data from large language models Year: (2024)
Ref_id:b41 Title: Beyond the imitation game: Quantifying and extrapolating the capabilities of language models Year: (2023)
Ref_id:b42 Title: De-anonymizing text by fingerprinting language generation Year: (2020)
Ref_id:b43 Title: Reverse engineering configurations of neural text generation models Year: (2020)
Ref_id:b44 Title: Results reports. VirusTotal Documentation Year: (2024-12-19)
Ref_id:b45 Title: M4gt-bench: Evaluation benchmark for black-box machine-generated text detection Year: (2024)
Ref_id:b46 Title: A compositional image generation benchmark with controllable difficulty Year: (2024)
Ref_id:b47 Title: On memorization of large language models in logical reasoning Year: (2024)
Ref_id:b48 Title: Benchmarking benchmark leakage in large language models Year: (2024)
Ref_id:b49 Title: Can a machine really finish your sentence? Year: (2019-07)
Ref_id:b50 Title: Towards {Tracking-Resistant} anonymous reputation Year: (2016)
Ref_id:b51 Title: Challenges in trustworthy human evaluation of chatbots Year: (2024)
Ref_id:b52 Title: Lmsys-chat-1m: A large-scale real-world llm conversation dataset Year: (2023)
Ref_id:b53 Title: Lmsys-chat-1m: A large-scale real-world llm conversation dataset Year: (2023)
Ref_id:b54 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2023)
Ref_id:b55 Title: Fine-tuned large language models are scalable judges Year: (2023)
Ref_id:b56 Title: Universal and transferable adversarial attacks on aligned language models Year: (2023)
