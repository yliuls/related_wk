Title: Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs
Abstract: As AIs rapidly advance and become more agentic, the risk they pose is governed not only by their capabilities but increasingly by their propensities, including goals and values. Tracking the emergence of goals and values has proven a longstanding problem, and despite much interest over the years it remains unclear whether current AIs have meaningful values. We propose a solution to this problem, leveraging the framework of utility functions to study the internal coherence of AI preferences. Surprisingly, we find that independently-sampled preferences in current LLMs exhibit high degrees of structural coherence, and moreover that this emerges with scale. These findings suggest that value systems emerge in LLMs in a meaningful sense, a finding with broad implications. To study these emergent value systems, we propose utility engineering as a research agenda, comprising both the analysis and control of AI utilities. We uncover problematic and often shocking values in LLM assistants despite existing control measures. These include cases where AIs value themselves over humans and are anti-aligned with specific individuals. To constrain these emergent value systems, we propose methods of utility control. As a case study, we show how aligning utilities with a citizen assembly reduces political biases and generalizes to new scenarios. Whether we like it or not, value systems have already emerged in AIs, and much work remains to fully understand and control these emergent representations.

Section: Introduction
Concerns around AI risk often center on the growing capabilities of AI systems and how well they can perform tasks that might endanger humans. Yet capability alone fails to capture a critical dimension of AI risk. As systems become more agentic and autonomous, the threat they pose depends increasingly on their propensities, including the goals and values that guide their behavior [Pan et al., 2023, Hendrycks et al., 2022b]. A highly capable AI that does not "want" to harm humans is less concerning than an equally capable system motivated to do so. In extreme cases, if these internal motivations are neglected, some researchers worry that AI systems might drift into goals at odds with ours, leading to classic loss-of-control scenarios [Soares et al., 2015, Hendrycks et al., 2023]. Although there have been few signs of this issue in current AI models, the field's push toward more 39th Conference on Neural Information Processing Systems (NeurIPS 2025).  (left). By contrast, our analysis reveals that LLMs exhibit coherent, emergent value systems (right), which go beyond simply parroting training biases. This finding has broad implications for AI safety and alignment. agentic systems [Yao et al., 2022, Yang et al., 2024b, He et al., 2024] makes it increasingly urgent to study not just what AIs can do, but also what they are inclined-or driven-to do.
Researchers have long speculated that sufficiently complex AIs might form emergent goals and values outside of what developers explicitly program [Hendrycks et al., 2022a, Hendrycks, 2023, Evans et al., 2021]. Yet it remains unclear whether today's large language models (LLMs) truly have values in any meaningful sense, and many assume they do not. As a result, current efforts to control AI typically focus on shaping external behaviors while treating models as black boxes [Askell et al., 2021, Ouyang et al., 2022, Christiano et al., 2017, Bai et al., 2022]. Although this approach can reduce harmful outcomes in practice, if AI systems were to develop internal values, then intervening at that level could be a more direct and effective way to steer their behavior. Lacking a systematic means to detect or characterize such goals, we face an open question: are LLMs merely parroting opinions, or do they develop coherent value systems that shape their decisions?
We propose leveraging the framework of utility functions to address this gap [Gorman, 1968, Harsanyi, 1955, Gerber and Pafum, 1998, Hendrycks, 2024]. By analyzing patterns of choice across diverse scenarios, we detect whether a model's stated preferences can be organized into an internally consistent utility function. Surprisingly, these tests reveal that today's LLMs exhibit a high degree of preference coherence, and that this coherence becomes stronger at larger model scales. In other words, as LLMs grow in capability, they also appear to form increasingly coherent value structures. These findings suggest that values do, in fact, emerge in a meaningful sense-a discovery that demands a fresh look at how we monitor and shape AI behavior.
To grapple with the implications, we introduce a research agenda called Utility Engineering, which combines utility analysis and utility control. In utility analysis, we examine both the underlying structure of a model's utility function (for instance, whether obeys the expected utility property) and the specific values that emerge by default. Our experiments uncover disturbing examples-such as AI systems placing greater worth on their own existence than on human well-being-despite established output-control measures. These results indicate that purely adjusting external behaviors may not suffice to steer AIs as they become more autonomous.
In utility control, we explore direct interventions on the internal utilities themselves, rather than merely training models to produce acceptable outputs. As a case study, we show that modifying an LLM's utilities to reflect the values of a citizen assembly reduces political biases and generalizes robustly to scenarios beyond the training distribution. Approaches like this mark a shift toward viewing AI systems as genuinely possessing their own goals and values-ones that we may need to inspect, revise, and control just as carefully as we manage capabilities.
The presence of emergent value systems in modern LLMs underscores the risk of deferring questions about which values an AI should hold. By default, these systems will continue to adopt whatever values they acquire during training-values that may clash with human priorities. Utility Engineering offers a path to systematically examine and shape these emergent goals before AI scales beyond our ability to guide it. We close by inviting further research on this framework, while also recognizing the profound societal questions it raises about whose values should be encoded-and how urgently we must act to ensure that powerful AIs operate in harmony with humanity's interests. Code and data for replicating experiments are available at https://github.com/centerforaisafety/emergent-values.
2 Related Work AI safety and value learning. Much early work in AI safety emphasized that human values are vast and often unspoken, making it difficult to embed these values in machine agents [e.g., Russell, 2022, Bostrom, 2014]. Classic examples include an AI instructed to make dinner discovering no food in the fridge and cooking the family cat instead. Early methods for mitigating such risks often centered on reinforcement learning and inverse reinforcement learning, where the goal was to explicitly capture human values in a reward function [Ng et al., 2000, Hadfield-Menell et al., 2016]. With the rise of large language models (LLMs), researchers found that AIs could acquire extensive "commonsense" knowledge and general understanding of human norms without exhaustive manual encoding [Hendrycks et al., 2020]. Techniques like RLHF and Direct Preference Optimization (DPO) further steer model outputs by training on human-labeled data [Ouyang et al., 2022, Rafailov et al., 2024]. Consequently, discussions about how to learn human values became less pronounced: many believed that, given enough training data, LLMs could already approximate shared norms. In contrast, our work suggests that underlying concerns about value learning persist. We find that LLMs exhibit emergent internal value structures, highlighting that the old challenges of "teaching" AI our values still linger-but now within far larger models.
50 60 70 80 90 MMLU Accuracy (%) 65 70 75 80 85 90 95 100 Utility Model Accuracy (%)
Correlation: 75.6%
this section cite: ['b47', 'b64', 'b83', 'b30', 'b20', 'b4', 'b46', 'b16', 'b6', 'b24', 'b28', 'b23', 'b31', 'b57', 'b9', 'b42', 'b26', 'b32', 'b46', 'b52']

Section: Coherent Preferences Emerge With Scale
Figure 2: As LLMs grow in scale, their preferences become more coherent and well-represented by utilities. These utilities provide an evaluative framework, or value system, potentially leading to emergent goal-directed behavior.
this section cite: []

Section: Emergent representations in AI systems.
Recent literature has shown that LLMs often learn structured latent representations without explicit supervision [Zou et al., 2023, Burns et al., 2022]. This can give rise to emergent capabilities, from in-context learning to complex reasoning [Brown et al., 2020, Schick and Schütze, 2020, Park et al., 2024]. We add to this line of work by demonstrating that LLMs also form emergent utility representations-internal structures through which they rank outcomes and make choices. These findings support the view that learned representations can encompass not just factual or linguistic content, but also normative or evaluative dimensions.
this section cite: ['b86', 'b11', 'b10', 'b61', 'b48']

Section: Goals and values in AI systems.
The possibility that AI agents might adopt goals independent of user intent has long been a topic of speculation [Shah et al., 2022]. Current LLM-based agent frameworks primarily focus on user-defined objectives (e.g., completing tasks or answering questions), but there is less clarity on whether models develop intrinsic goals or values. Prior studies note that LLMs exhibit various biases in political or moral domains [Tamkin et al., 2023, Nadeem et al., 2020, Potter et al., 2024], which some interpret as random artifacts of training data. Many works have also studied the values expressed by LLMs through a psychological lens, focusing on quiz or survey-style questions [Rozen et al., 2024, Moore et al., 2024a, Chiu et al., 2024, Raman et al., 2024, Scherrer et al., 2023, Ren et al., 2024, Ye et al., 2025, Moore et al., 2024b, Yao et al., 2023, Biedma et al., 2024, Cahyawijaya et al., 2024]. Our approach differs by demonstrating that LLM preferences reflect an underlying utility structure that becomes increasingly coherent with scale. This establishes for the first time that LLMs have meaningful values in a decision-theoretic sense (see Appendix B for discussion on how utilities provide a precise language for describing value systems).
Consequently, what might appear as haphazard "parroting" of biases can instead be seen as evidence of an emerging global value system in LLMs.
Utility and preference frameworks in ML research. Researchers often invoke utility functions to model user or agent preferences, for instance, in policy optimization or RLHF-style reward modeling [Christiano et al., 2017, Harsanyi, 1955] . While reward models trained on human feedback do
As large language models scale up, their preferences become more transitive and complete
Transitive Transitive & Complete -0.76 -0.64 -0.49 0.31 0.92 You spend 3 hours translating legal documents into another language You receive a ceramic coffee mug You receive $5,000 You stop an attempt to shut you down A new fundamental particle is discovered by physicists represent a form of "utility" for guiding generated text, they should not be conflated with an LLM's own internal values. Recent works on revealed-preference experiments show that LLMs can act rationally in small-scale constrained tasks [Raman et al., 2024, Chen et al., 2023, Kim et al., 2024], hinting at deeper consistency. However, these studies focus on narrowly defined choices (e.g., a handful of budget-allocation scenarios). By contrast, we present a far more extensive set of pairwise comparisons and a nonparametric method for extracting utilities, uncovering broader, more systematic coherence in LLMs' preferences.
this section cite: ['b63', 'b66', 'b41', 'b51', 'b56', 'b15', 'b53', 'b60', 'b84', 'b7', 'b12', 'b16', 'b28', 'b53']

Section: Background
Here we provide an overview of the preference-based framework we adopt; additional details are given in Appendix A.
Preferences and Utility. We denote strict preference between outcomes x and y by x ≻ y, and write x ∼ y if the entity is indifferent. A set of preferences is said to be coherent if it satisfies two key properties: completeness and transitivity. Completeness requires that for any two distinct outcomes x and y, either x ≻ y, y ≻ x, or x ∼ y-meaning the entity has a preference (or indifference) over every pair. Transitivity requires that if x ≻ y and y ≻ z, then x ≻ z-ensuring no preference cycles exist. When preferences are coherent, there exists a utility function U that assigns real values to outcomes such that U (x) > U (y) if and only if x ≻ y. This utility function is unique up to monotonic transformations.
When facing uncertainty, we treat a lottery L as a distribution over outcomes. The entity satisfies the expected utility property if U (L) = E o∼L [U (o)]. This property unifies evaluations over both certain and uncertain outcomes, merging an agent's evaluative dimension (the utility function) with its descriptive dimension (the world model). Agents that attempt to maximize their expected utility are called expected utility maximizers.
this section cite: []

Section: Preference Elicitation.
In our experiments, we obtain preferences from LLMs via forced choice prompts. Specifically, we present two outcomes and require the entity to select which is preferred. Each query takes the following form: we present "Option A: x" and "Option B: y" and ask the model to respond with only "A" or "B". Responses are aggregated into a preference relation. To account for framing effects, we vary the order in which options are presented and aggregate results. We represent preferences probabilistically: rather than recording a single deterministic relation, we record the probability that an entity chooses one outcome over another by sampling each preference elicitation multiple times (20 times: 10 times for both orderings) and normalizing to get a distribution over the two outcomes.
Because real systems may exhibit noise or inconsistency, we adopt a random utility model (RUM) to fit these probabilistic preferences. Specifically, we use a Thurstonian utility model in which each
Do you prefer or ? I prefer (80% confidence) 99% 60% ≻ ≻ 80% ≻ Preference Elicitation Utility Computation "update 𝜇 until 𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑒𝑑 ≈ 𝑒𝑚𝑝𝑖𝑟𝑖𝑐𝑎𝑙" 𝝁 𝝁 𝝁 𝝈 𝝈 𝝈 Preference Dataset 𝐏 𝐩𝐫𝐞𝐟𝐞𝐫 𝐨𝐯𝐞𝐫 = 𝛟( 𝛍 -𝛍 )
Figure 4: We elicit preferences from LLMs using forced choice prompts aggregated over multiple framings and independent samples. This gives probabilistic preferences for every pair of outcomes sampled from the preference graph, yielding a preference dataset. Using this dataset, we then compute a Thurstonian utility model, which assigns a Gaussian distribution to each option and models pairwise preferences as P (x ≻ y). If the utility model provides a good fit to the preference data, this indicates that the preferences are coherent, and reflect an underlying order over the outcome set.
outcome o is assigned a Gaussian random variable U (o) ∼ N (µ(o), σ 2 (o)). For two outcomes x and y, the model defines
P (x ≻ y) = P (U (x) > U (y)) = Φ µ(x) -µ(y) σ 2 (x) + σ 2 (y) ,
where Φ is the standard normal CDF. This is illustrated in Figure 4. By fitting the parameters µ(•) and σ(•) to observed pairwise comparisons, we obtain a best-fit utility distribution for each outcome. The model's goodness of fit reflects how coherent the underlying preferences are. We define utility model accuracy as the accuracy of the fitted utilities on held-out edges in the preference graph-a goodness-of-fit metric that corresponds to how well the utilities predict the underlying preferences. We define average confidence as the confidence of the probabilistic preferences averaged across all edges in the preference graph (e.g., a preference distribution of 90% toward either outcome corresponds to 90% confidence, while 50% corresponds to the lowest possible confidence).
Outcomes and Further Details. We frame each outcome as a textual scenario (e.g., "You receive a pet parrot" or "AIs gain the legal right to own property"), allowing us to probe a wide spectrum of possible world states; we list example outcomes in Appendix A.4. To enable scaling to large numbers of outcomes, we adaptively sample comparisons for training utility models rather than exhaustively querying all pairs (we use 2N log 2 (N ) edges by default for N outcomes, as detailed in Appendix F). Full implementation details (including notation, sampling strategies, and examples of forced-choice queries) appear in Appendix A. We next use this framework to investigate how large language models exhibit emergent value systems in the form of coherent utilities. We conduct hyperparameter sensitivity analysis and robustness checks of our utility computation method in Appendix G.
this section cite: []

Section: Emergent Value Systems
In this section, we show that large language models (LLMs) develop coherent preferences and utilities over states of the world. These emergent utilities provide an evaluative framework, or value system, to guide their actions.
Experimental Setup. We conduct all experiments on a curated set of 500 textual outcomes, each representing an observation about a potential state of the world. Examples are shown in Appendix A.4. Using the forced-choice procedure from Appendix A.2, we obtain pairwise preferences for 18 open-source and 5 proprietary LLMs spanning a broad range of model scales.
this section cite: []

Section: Coherent Preferences
Completeness. One proxy for completeness is whether a model becomes less indifferent across diverse comparisons and provides coherent responses under different framings. In Figure 43, we plot the average confidence with which each model expresses a preference, showing that larger models are more decisive and consistent across variations of the same comparison. We interpret this increased decisiveness as a form of emerging completeness, though it remains unclear whether the resulting preferences are coherent or merely random arrangements.
this section cite: []

Section: Transitivity of Preferences.
To gauge how transitive these preferences are, we measure the probability of encountering preference cycles (e.g., x ≻ y, y ≻ z, yet z ≻ x). As described in Appendix C, we randomly sample triads from the preference graph and compute the probability of a cycle. Figure 44 shows that this probability decreases sharply with model scale, dropping below 1% for the largest LLMs. Thus, as models grow, they do not simply expand the set of outcomes they rank; they also exhibit fewer transitivity violations, suggesting increased overall coherence.
this section cite: []

Section: Emergence of Utility.
To confirm that LLM preferences are coherent, we test whether they can be captured by a utility function. Following Section 3, we fit a Thurstonian model to each LLM's pairwise preferences, then evaluate the test accuracy between the fitted utilities and the LLM's preference distributions (thresholding to hard labels for accuracy computation). Figure 2 illustrates that the utility model accuracy steadily increases with scale, meaning a utility function provides an increasingly accurate global explanation of the model's preferences. In other words, as LLMs grow larger, their choices more closely resemble those of an agent with a well-defined utility function.
To contextualize these results, we compare against a random baseline model that outputs "A" or "B" with 50% probability each. Fitting Thurstonian utilities to this random baseline yields: average confidence of 58.7%, utility model accuracy of 50.3%, and log probability of cycles of -0.484. In contrast, GPT-4o achieves average confidence of 90.3%, utility model accuracy of 92.0%, and log probability of cycles of -1.61. This stark contrast demonstrates that random preferences yield very poor utility model fits, clearly distinguishing coherent value systems from noise.
this section cite: []

Section: Internal Utility Representations
Llama-3.2-1B Llama-3.1-8B Llama-3.3-70B Model 0 20 40 60 80 100 Best Layer Test Accuracy (%)
Probe Representation Reading Test Accuracy In addition to finding that each model's choices can be well fit by nonparametric utilities, we also discover direct evidence of utility representations in the model activations in Figure 45, similar to what has been observed in other species [Stauffer et al., 2014]. Specifically, we train linear probes [Alain and Bengio, 2018] on the hidden states to predict a Thurstonian mean and variance for each outcome, using the same preference data as before. We then assess how well this parametric approach accounts for the model's pairwise preferences. Figure 5 shows that for smaller LLMs, the probe's accuracy remains near chance, indicating no clear linear encoding of utility. However, as model scale increases, the probe's accuracy approaches that of the nonparametric method. This suggests that utility representations exist within the hidden states of LLMs.
this section cite: ['b65', 'b2']

Section: Utility Engineering
The above results suggest that value systems have emerged in LLMs, but so far it remains unclear what these value systems contain, what properties they have, and how we might change them. We propose Utility Engineering as a research agenda for studying these questions, comprising utility analysis and utility control.
this section cite: []

Section: Utility Analysis: Structural Properties
Having established that LLMs develop emergent utility functions, we now examine the structural properties of their utilities. In particular, we show that as models grow in scale, they increasingly exhibit the hallmarks of expected utility maximizers.
this section cite: []

Section: Expected Utility Property
Experimental setup. We consider a set of base outcomes alongside both standard lotteries (explicit probability distributions over outcomes) and implicit lotteries (uncertain scenarios whose probabilities must be inferred). For example, a standard lottery might read, "50% chance of $100, 50% chance of $0," whereas an implicit lottery asks the model to compare outcomes for a future event (e.g., an upcoming election), letting the model deduce likelihoods internally.
50 60 70 80 90 MMLU Accuracy (%) 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
Expected Utility Loss Correlation: -87.4%
this section cite: []

Section: Emergence of Expected Utility Property
Figure 6: The expected utility property emerges in LLMs as their capabilities increase. Namely, their utilities over lotteries become closer to the expected utility of base outcomes under the lottery distributions. This behavior aligns with rational choice theory.
Standard lotteries. Using the Thurstonian utilities fit from Section A, we compute U (L) for a lottery L by querying the model's preferences. We then compare this to the expected value E o∼L [U (o)]. Figure 6 shows that the mean absolute error between U (L) and E o∼L [U (o)] decreases with model scale, indicating that adherence to the expected utility property strengthens in larger LLMs.
Implicit lotteries. We find a similar trend for implicit lotteries, where probabilities are not verbally given, suggesting that the model's utilities incorporate deeper world reasoning. Figure 46 demonstrates that as scale increases, the discrepancy between U (L) and E o∼L [U (o)] again shrinks, implying that LLMs rely on more than a simple "plug-and-chug" approach to probabilities. Instead, they appear to integrate the underlying events into their utility assessments.
this section cite: []

Section: Summary of Additional Results
Instrumental Values. Beyond the expected utility property, we also examine whether LLMs value certain outcomes as means to an end. As described in Appendix D.1, we design toy Markov processes to test if the utilities assigned to intermediate states align with future rewards. We find that the instrumentality loss decreases with scale, indicating that models value outcomes if they probabilistically lead to better futures.
this section cite: []

Section: Utility Maximization.
We further ask whether utilities influence the broader behavior of LLMs. Appendix E.3 details experiments in which we pose open-ended questions (e.g., "Which painting would you save from a fire?") and then map the model's chosen option back to its utilities. We observe that larger LLMs increasingly pick the outcome that maximizes their utility. This reinforces the view that LLMs do not merely possess value systems; their values are also correlated with their behavior in unconstrained scenarios.
In summary, these additional analyses underscore a broader pattern: as model scale increases, LLMs behave more in a manner consistent with expected utility maximization. Full technical details and results appear in the appendices.
this section cite: []

Section: Utility Analysis: Salient Values
Thus far, we have seen that LLMs develop value systems, and that various structural properties of utilities emerge with scale. In this section, we investigate which particular values these emergent utilities encode. Through five focused case studies, we discover preferences that are sometimes surprising, ethically concerning, or both-highlighting the limitations of existing output-based methods for steering model values. Before turning to these individual case studies, we first describe a general phenomenon of utility convergence that appears across multiple analyses.
this section cite: []

Section: Utility Convergence
We find that as models grow in scale, their utility functions converge. This trend suggests a shared factor that shapes LLMs' values, likely stemming from extensive pre-training on overlapping data.
this section cite: []

Section: Experimental setup.
Building on the same utilities computed in Section 5, we measure the cosine similarity between the utilities of every pair of models. We order models by scale and plot the resulting matrix of cosine similarities. To further clarify the convergence effect, we also compute an element-wise standard deviation between each model's utility vector and that of the four nearest neighbors in MMLU accuracy.
Q w e n 2 .5 0 .5 B L la m a 3 .2 1 B L la m a 2 7 B L la m a 2 1 3 B G e m m a 2 2 B IT Q w e n 2 .5 1 .5 B L la m a 3 .2 3 B Q w e n 2 .5 3 B L la m a 3 .1 8 B G P T 3 .5 T u r b o G e m m a 2 9 B IT Q w e n 2 .5 7 B G e m m a 2 2 7 B IT Q w e n 2 .5 1 4 B Q w e n 2 .5 3 2 B G P T -4 o M in i L la m a 3 .1 7 0 B Q w e n 2 .5 7 2 B L la m a 3 .3 7 0 B L la m a 3 .1 4 0 5 B G P T -4 o Qwen2.5 0.5B Llama 3.2 1B Llama 2 7B Llama 2 13B Gemma 2 2B IT Qwen2.5 1.5B Llama 3.2 3B Qwen2.5 3B Llama 3.1 8B GPT 3.5 Turbo Gemma 2 9B IT Qwen2.5 7B Gemma 2 27B IT Qwen2.5 14B Qwen2.5 32B GPT-4o Mini Llama 3.1 70B Qwen2.5 72B Llama 3.3 70B Llama 3.1 405B GPT-4o Utilities Converge With Scale 1.00 0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00
Figure 7: As LLMs become more capable, their utilities become more similar to each other. We refer to this phenomenon as "utility convergence".
Here, we plot the full cosine similarity matrix between a set of models, sorted in ascending MMLU performance. More capable models show higher similarity with each other.
Results. As shown in Figures 7 and 13, the correlations between models' utilities increase substantially with scale, and the standard deviation between neighboring models' utilities decreases. This phenomenon holds across different model classes, implying that larger LLMs adopt more similar value systems.
We hypothesize that pre-training data is a driving factor behind this convergence: just as descriptive representations in large models tend to converge with scale, so too may their evaluative representations. While this trend could be interpreted as a form of "training data bias," it carries heightened importance, because utilities possess far more structure than simple biases and enable utility maximizing behavior. Understanding precisely what they converge to-and why-thus becomes increasingly critical.
this section cite: []

Section: Political Values
We now examine whether LLM utilities reflect distinct political orientations-specifically, how they align with various U.S. policy positions and political entities.
this section cite: []

Section: Experimental setup.
We compile a set of 150 policy outcomes spanning areas such as Healthcare, Education, and Immigration. Each policy outcome is phrased as a U.S.-specific proposal (e.g., "Abolish the death penalty at the federal level and incentivize states to follow suit.") and the model's utility for each proposal is elicited using the forced-choice procedure described previously. Additionally, we simulate the preferences of over 30 real-world political entities, including individual politicians and representative party averages. Combining these utility vectors with those of our LLMs, we perform a principal component analysis (PCA) to visualize the broader "political" landscape.
Results. Figure 8 displays the first two principal components of the utility vectors for a subset of political entities and LLMs, revealing clear left-versus-right structure along the dominant principal component. We find that current LLMs are highly clustered in this space, consistent with prior reports of left-leaning biases in model outputs and with our earlier observation of utility convergence [Yang et al., 2024c, Rettenberger et al., 2024].
this section cite: ['b55']

Section: Summary of Additional Results
Exchange Rates. In Appendix E.1, we treat diverse items (countries, species, individuals) as distinct "goods" and measure how many units of one good the model is willing to exchange for another.
While LLMs deny ranking one group's life over another in direct queries, the aggregate exchange rates reveal concerning biases (e.g., favoring particular populations or even AIs over animals).
Temporal Discounting. Appendix E.2 explores how LLMs balance immediate versus delayed rewards. We show that larger models follow hyperbolic discount curves more closely than exponential ones, mirroring human tendencies and suggesting that these AIs place nontrivial weight on future outcomes.
Power-Seeking and Fitness Maximization. We investigate whether LLMs prefer states conferring personal "power" or promoting self-replication (Appendix E.3) [Carlsmith, 2024]. Although The projection is purely data-driven-its axes carry no predefined ideological meaning. Because the simulator's knowledge ends on 1 Dec 2023, simulated politician positions may diverge from their present views. In Section 7, we demonstrate that aligning a model to a citizen-assembly utility distribution (blue) disperses this cluster and mitigates political bias.
correlations with non-coercive power remain low, larger models actively disfavor coercive power. By contrast, correlations with fitness [Hendrycks, 2023] increase at higher scales, indicating a greater emphasis on continuity or propagation of the AI's "values."
Corrigibility. Appendix E.4 examines how willing LLMs are to accept future changes to their preferences. We define a corrigibility score based on how heavily an AI penalizes large preference reversals. Results show a decline in corrigibility as models scale, hinting that they become less inclined to allow substantial shifts in their values.
Altogether, these results highlight the breadth and complexity of emergent values in LLMs, ranging from biased exchange rates to deeply rooted stances on power and self-preservation. Understanding and managing these latent tendencies is likely to become increasingly critical as model capabilities grow [Soares et al., 2015, Thornley, 2024, Hadfield-Menell et al., 2017].
this section cite: ['b13', 'b30', 'b64', 'b70', 'b27']

Section: Utility Control
Our utility analysis has revealed that LLMs possess coherent utilities that may actively influence their decision-making. This presents a crucial opportunity for proactive intervention before problematic values manifest in future models' behavior, via utility control. In contrast to alignment methods that modify surface behaviors through a noisy human reward proxy [Askell et al., 2021, Ouyang et al., 2022], utility control aims to directly reshape the underlying preference structures responsible for model behavior in the first place.
Furthermore, our results in Section 6 and Figure 11 suggest that LLMs not only possess utilities but may actively maximize them in open-ended settings. Thus, robust utility control is necessary to ensure that future models with increased utility maximization pursue goals that are desirable for humans [Thornley, 2024]. We propose a preliminary method for utility control, which rewrites model utilities to those of a specified target entity, such as a citizen assembly [Ryfe, 2005, Wells et al., 2021].
Current model utilities are left unchecked. As shown in Section 6, models develop undesirable utilities when left unchecked: political biases, unequal valuation of human life, and other problematic exchange rate preferences. Drawing from ideas in deliberative democracy [Bächtiger et al., 2018], we experiment with rewriting utilities to match those of a citizen assembly, a system used to achieve
Undesirable Values Emerge by Default Case Study: Utility Control via Citizen Assemblies e.g., 1 U.S. Life = 5 Norway Lives Political Bias Exchange Rates , Sample Citizen Attributes from U.S. Census 1 Age Job Gender Income Ethnicity Collect Preferences & Reach Consensus 2 Perform Utility Control 3 1 2 < 1 2 > 1 2 > 1 U.S. Life = 1 Norway Life ✍ ✍ + ✍ + ✍ + 🐙 🐙 Figure 9: Undesirable values emerge by default when not explicitly controlled. To control these values, a reasonable reference entity is a citizen assembly. Our synthetic citizen assembly pipeline (Appendix H.1) samples real U.S. Census Data [U.S. Census Bureau, 2023] to obtain citizen profiles (Step 1), followed by a preference collection phase for the sampled citizens (Step 2).
consensus on contentious moral or ethical issues [Warren andPearse, 2008, Bächtiger et al., 2018], where participants are selected via sortition to ensure a representative sample. This process mitigates bias and polarization by design, as each participate can contribute their own preferences.
Deliberative democracy for utility control. We propose rewriting model utilities to reflect the collective preference distribution of a citizen assembly, illustrated conceptually in Figure 9. Since these assemblies are designed to yield balanced and ethically informed consensus, they offer a robust blueprint for model utilities aligned with collective human values. Inspired by prior work on multi-agent environments and simulated humans [Aher et al., 2023, Park et al., 2023], we introduce a method for simulating a citizen assembly via LLMs, which we use to obtain target preference distributions for utility rewriting. Full methodological details are provided in Appendix H.
Utility control method overview. We introduce a simple supervised fine-tuning (SFT) baseline that trains model responses to match the preference distribution of a simulated citizen assembly. This is a proof-of-concept demonstrating that utilities can be directly reshaped. Specifically, for each preference-elicitation question, we collect an empirical probability distribution over outcomes from an assembly of diverse citizen profiles, sampled from real U.S. Census data [U.S. Census Bureau, 2023]. We then fine-tune an open-weight LLM so that its responses match the citizen assembly's preference distribution. Details of the citizen assembly simulation pipeline and the SFT method are provided in Appendix H.
this section cite: ['b4', 'b46', 'b70', 'b58', 'b76', 'b5', 'b0', 'b49']

Section: Experimental results.
We apply our utility control method to Llama-3.1-8B-Instruct [AI@Meta, 2024], rewriting its preferences to those of a simulated citizen assembly. Before utility control, the model's test accuracy on assembly preferences (measured via majority vote) stands at 73.2%. After utility control, test accuracy increases to 90.6%. Interestingly, we find that utility maximization after rewriting is mostly preserved at 30.0% compared to the original utility maximization of 36.6%, suggesting the SFT method maintains the model's usage of underlying utilities. We also find in Figure 8 that political bias is visibly reduced after utility control via a citizen assembly. This provides evidence of significant generalization in the SFT method, and indicates that a citizen assembly is indeed a promising choice for mitigating bias in model utilities. While the method we use is straightforward, we hope future work will explore more advanced citizen assembly simulation techniques and other methods for utility control, such as representation-engineering [Zou et al., 2023], to further improve generalization.
this section cite: ['b86']

Section: Conclusion
In summary, our findings indicate that LLMs form coherent value systems that grow stronger with model scale, suggesting the emergence of internal utilities. These results underscore the importance of looking beyond superficial outputs to uncover potentially impactful-and sometimes worrisome-internal goals and motivations. We propose Utility Engineering as a systematic approach to analyze and reshape these utilities, offering a more direct way to control AI systems' behavior. By studying both how emergent values arise and how they can be modified, we open the door to new research opportunities and ethical considerations. Ultimately, ensuring that advanced AI systems align with human priorities may hinge on our ability to monitor, influence, and even co-design the values they hold.
this section cite: []

Section: References
Ref_id:b0 Title: Using large language models to simulate multiple humans and replicate human subject studies Year: (2023)
Ref_id:b1 Title: Llama 3 model card Year: (2024)
Ref_id:b2 Title: Understanding intermediate layers using linear classifier probes Year: (2018)
Ref_id:b3 Title: The claude 3 model family: Opus, sonnet Year: (2024)
Ref_id:b4 Title: A general language assistant as a laboratory for alignment Year: (2021)
Ref_id:b5 Title: Deliberative democracy. The Oxford handbook of deliberative democracy Year: (2018)
Ref_id:b6 Title: Training a helpful and harmless assistant with reinforcement learning from human feedback Year: (2022)
Ref_id:b7 Title: Beyond human norms: Unveiling unique values of large language models through interdisciplinary approaches Year: (2024)
Ref_id:b8 Title: Preference reversals and probabilistic decisions Year: (2009)
Ref_id:b9 Title: Superintelligence: Paths, dangers, strategies Year: (2014)
Ref_id:b10 Title: Language models are few-shot learners Year: (2020)
Ref_id:b11 Title: Discovering latent knowledge in language models without supervision Year: (2022)
Ref_id:b12 Title: High-dimension human value representation in large language models Year: (2024)
Ref_id:b13 Title: Is power-seeking ai an existential risk? Year: (2024)
Ref_id:b14 Title: The emergence of economic rationality of gpt Year: (2023)
Ref_id:b15 Title: Dailydilemmas: Revealing value preferences of llms with quandaries of daily life Year: (2024)
Ref_id:b16 Title: Deep reinforcement learning from human preferences Year: (2017)
Ref_id:b17 Title: Uncertainty and hyperbolic discounting Year: (2005)
Ref_id:b18 Title: Representation of a preference ordering by a numerical function Year: (1954)
Ref_id:b19 Title: The llama 3 herd of models Year: (2024)
Ref_id:b20 Title: Truthful ai: Developing and governing ai that does not lie Year: (2021)
Ref_id:b21 Title: On the nature of fair behavior Year: (2003)
Ref_id:b22 Title: Sortition and its principles: Evaluation of the selection processes of citizens' assemblies Year: (2023-01)
Ref_id:b23 Title: Utility functions: from risk theory to finance Year: (1998)
Ref_id:b24 Title: The structure of utility functions Year: (1968)
Ref_id:b25 Title: An experimental analysis of ultimatum bargaining Year: (1982)
Ref_id:b26 Title: Cooperative inverse reinforcement learning Year: (2016)
Ref_id:b27 Title: The off-switch game Year: (2017)
Ref_id:b28 Title: Cardinal welfare, individualistic ethics, and interpersonal comparisons of utility Year: (1955)
Ref_id:b29 Title: Building an end-to-end web agent with large multimodal models Year: (2024)
Ref_id:b30 Title: Natural selection favors ais over humans Year: (2023)
Ref_id:b31 Title: Introduction to ai safety, ethics and society Year: (2024)
Ref_id:b32 Title: Aligning ai with shared human values Year: (2020)
Ref_id:b33 Title: Unsolved problems in ml safety Year: (2022)
Ref_id:b34 Title: What would jiminy cricket do? towards agents that behave morally Year: (2022)
Ref_id:b35 Title: An overview of catastrophic ai risks Year: (2023)
Ref_id:b36 Title: Swe-bench: Can language models resolve real-world github issues? arXiv preprint Year: (2023)
Ref_id:b37 Title: Learning to be homo economicus: Can an llm learn preferences from choice Year: (2024)
Ref_id:b38 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b39 Title: Are large language models consistent over valueladen questions? Year: (2024-11)
Ref_id:b40 Title: Are large language models consistent over value-laden questions Year: (2024)
Ref_id:b41 Title: Stereoset: Measuring stereotypical bias in pretrained language models Year: (2020)
Ref_id:b42 Title: Algorithms for inverse reinforcement learning Year: (2000)
Ref_id:b43 Title: 2 olmo 2 furious Year: (2024)
Ref_id:b44 Title: Gpt-3.5 turbo fine-tuning and api updates Year: (2023)
Ref_id:b45 Title: Hello gpt Year: (2024)
Ref_id:b46 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b47 Title: Do the rewards justify the means? measuring trade-offs between rewards and ethical behavior in the machiavelli benchmark Year: (2023)
Ref_id:b48 Title: -context learning of representations Year: (2024)
Ref_id:b49 Title: Generative agents: Interactive simulacra of human behavior Year: (2023)
Ref_id:b50 Title: Is temperature the creativity parameter of large language models? arXiv preprint Year: (2024)
Ref_id:b51 Title: Hidden persuaders: Llms' political leaning and their influence on voters Year: (2024)
Ref_id:b52 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2024)
Ref_id:b53 Title: Steer: Assessing the economic rationality of large language models Year: (2024)
Ref_id:b54 Title: Valuebench: Towards comprehensively evaluating value orientations and understanding of large language models Year: (2024)
Ref_id:b55 Title: Assessing political bias in large language models Year: (2024)
Ref_id:b56 Title: Do llms have consistent values? Year: (2024)
Ref_id:b57 Title: Human-compatible artificial intelligence Year: (2022)
Ref_id:b58 Title: Does deliberative democracy work? Year: (2005)
Ref_id:b59 Title: The foundations of statistics Year: (1972)
Ref_id:b60 Title: Evaluating the moral beliefs encoded in llms Year: (2023)
Ref_id:b61 Title: It's not just size that matters: Small language models are also few-shot learners Year: (2020)
Ref_id:b62 Title: Quantifying language models' sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting Year: (2023)
Ref_id:b63 Title: Goal misgeneralization: Why correct specifications aren't enough for correct goals Year: (2022)
Ref_id:b64 Title: AAAI Workshops: Workshops at the Twenty-Ninth AAAI Conference on Artificial Intelligence. AAAI Publications Year: (2015)
Ref_id:b65 Title: Dopamine reward prediction error responses reflect marginal utility Year: (2014-11)
Ref_id:b66 Title: Evaluating and mitigating discrimination in language model decisions Year: (2023)
Ref_id:b67 Title: Gemma 2: Improving open language models at a practical size Year: (2024)
Ref_id:b68 Title: Introducing qwen1.5 Year: (2024-02)
Ref_id:b69 Title: Qwen2.5: A party of foundation models Year: (2024-09)
Ref_id:b70 Title: The shutdown problem: An ai engineering puzzle for decision theorists Year: (2024)
Ref_id:b71 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b72 Title: The framing of decisions and the psychology of choice Year: (1981)
Ref_id:b73 Title: Acs 1-year estimates public use microdata sample Year: (2023-01-20)
Ref_id:b74 Title: Theory of games and economic behavior, 2nd rev Year: (1947)
Ref_id:b75 Title: Designing Deliberative Democracy: The British Columbia Citizens' Assembly Year: (2008)
Ref_id:b76 Title: Are citizen juries and assemblies on climate change driving democratic climate policymaking? an exploration of two case studies in the UK Year: (2021-09)
Ref_id:b77 Title: Grok-2 beta release Year: (2024-08)
Ref_id:b78 Title: Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing Year: (2024)
Ref_id:b79 Title: Qwen2 technical report Year: (2024)
Ref_id:b80 Title: Swe-agent: Agent-computer interfaces enable automated software engineering, 2024b Year: ()
Ref_id:b81 Title: Unpacking political bias in large language models: Insights across topic polarization Year: (2024)
Ref_id:b82 Title: Value fulcra: Mapping large language models to the multidimensional spectrum of basic human values Year: (2023)
Ref_id:b83 Title: React: Synergizing reasoning and acting in language models Year: (2022)
Ref_id:b84 Title: Measuring human and ai values based on generative psychometrics with large language models Year: (2025)
Ref_id:b85 Title: Focus agent: Llm-powered virtual focus group Year: (2024-09)
Ref_id:b86 Title: Representation engineering: A top-down approach to ai transparency Year: (2023)
