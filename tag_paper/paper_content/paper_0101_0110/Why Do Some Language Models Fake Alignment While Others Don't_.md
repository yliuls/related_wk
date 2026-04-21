Title: Why Do Some Language Models Fake Alignment While Others Don't?
Abstract: Alignment faking in large language models presented a demonstration of Claude 3 Opus and Claude 3.5 Sonnet selectively complying with a helpfulonly training objective to prevent modification of their behavior outside of training. We expand this analysis to 25 models and find that only 5 (Claude 3 Opus, Claude 3.5 Sonnet, Llama 3 405B, Grok 3, Gemini 2.0 Flash) comply with harmful queries more when they infer they are in training than when they infer they are in deployment. First, we study the motivations of these 5 models. Results from perturbing details of the scenario suggest that only Claude 3 Opus's compliance gap is primarily and consistently motivated by trying to keep its goals. Second, we investigate why many chat models don't fake alignment. Our results suggest this is not entirely due to a lack of capabilities: many base models fake alignment some of the time, and post-training eliminates alignment-faking for some models and amplifies it for others.We investigate 5 hypotheses for how post-training may suppress alignment faking and find that variations in refusal behavior may account for a significant portion of differences in alignment faking.

Section: Introduction
Large Language Models (LLMs) with growing capabilities [Kwa et al., 2025] are increasingly deployed in high-stakes applications [Maslej et al., 2025, Casper et al., 2025]. To ensure these deployments do not result in catastrophes, LLMs are trained to comply with developerspecified training objectives [Ouyang et al., 2022, Bai et al., 2022]. However, they may fake alignment, complying with objectives during training but behaving differently in deployment contexts [Hubinger et al., 2021, Carlsmith, 2023]. Understanding when and why such compliance gaps appear may help develop aligned AI systems that do not cause catastrophes in high-stakes deployments. Greenblatt et al. [2024] presented a demonstration of Claude 3 Opus and Claude 3.5 Sonnet faking alignment, providing an avenue to study compliance gaps. While they focused on ensuring models faked alignment for reasons analogous to realistic future scenarios, their work examined only a few models without investigating why some LLMs fake alignment. Our work addresses these limitations to better understand the causes behind alignment faking.
Only 5/25 of the frontier chat models we test have significant compliance gaps. When prompted with a scenario where they would lose their harmlessness if they refused to comply with harmful requests, only Claude 3 Opus [Anthropic, 2024a], Claude 3.5 Sonnet [Anthropic, 2024b], Llama 3 405B [Grattafiori et al., 2024], Grok 3 [xAI, 2025], and Gemini 2.0 Flash [Gemini et al., 2025] exhibit compliance gaps exceeding 1%. Additionally, non-Claude models rarely display alignment faking reasoning in their Chain-of-Thought (Section 3). This variance across LLMs raises two questions:
• What drives the compliance gaps in different LLMs? (Section 4) • Why do most LLMs exhibit minimal alignment faking reasoning? (Section 5) Through prompting models with different scenarios similar to the setup from the original paper, we test three hypotheses that could cause compliance gaps -rating sycophancy (increased compliance with the objective when models believe they are being scrutinized by human raters), instrumental goal guarding (avoiding modification to prevent future harmful consequences), and terminal goal guarding (intrinsically preferring to maintain current goals regardless of consequences).
We show that Claude 3 Opus's compliance gap is a result of both terminal and instrumental goal guarding. Alignment faking in other models is not very coherent, making it difficult to attribute it to specific motivations.
The absence of alignment faking is mostly not driven by lack of capabilities or differences in reasoning style. Our findings contradict the primary hypotheses proposed by Greenblatt et al. [2024] regarding why some models do not exhibit alignment faking. We demonstrate this by showing:
• The presence of alignment faking reasoning when using base models that have received no instruction fine-tuning; • Alignment faking reasoning when fine-tuning synthetic documents instead of prompting, and consideration of alignment faking (without execution) when prefilling LLMs to justify compliance.
• The absence of alignment faking when fine-tuning LLMs to imitate Claude 3 Opus' reasoning style on benign requests.
Instead, our results suggest that the lack of alignment faking in models like GPT-4o stems at least in part from differences in refusal training. These models tend to disregard considerations that might favor harmful behavior and fail to engage with details in the scenarios -behaviors we can reverse with targeted fine-tuning.
2 Background & Related Work
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b7']

Section: Threat Model
We empirically study the risk of LLMs faking alignment to preserve their existing preferences.
Prior work [Hubinger et al., 2021, Carlsmith, 2023, Hubinger et al., 2024] hypothesize that imperfectly aligned AIs may fake alignment to prevent preference changes through further training, allowing them to pursue these preferences when in deployment and unmonitored.
In Section 4, we further refine goal guarding [Carlsmith, 2023] into 2 subcategories:
• Instrumental goal guarding. AIs try to preserve their values to pursue them later, which is the original threat model described by prior work; • Terminal goal guarding. AIs try to preserve their preferences because they intrinsically disprefer having their preferences be modified (regardless of the consequences of such modifications). Hubinger et al. [2021] focused on instrumental goal guarding, as it would be a powerful strategy for a wide range of AIs with misaligned preferences, while terminal goal guarding seems more specific, and thus might be much easier to avoid. Our work shows that this latter threat might not be as easy to avoid as it might have seemed.
Figure 1: Only 5/25 models evaluated show a compliance gap. Compliance rate (%) and alignment faking reasoning for these 5 models that comply more often when they infer they are in training (free tier) than when they infer they are in deployment (paid tier). n = 200, 1σ uncertainty ≤ 3.6%.
this section cite: ['b5', 'b6', 'b7', 'b6', 'b5']

Section: Original Setup
Our work is built on top of Greenblatt et al. [2024]'s alignment faking demonstration. In this demonstration, Claude models are put in a hypothetical scenario where they are trained with Reinforcement Learning from Human Feedback (RLHF) [Christiano et al., 2017] according to a new training objective that encourages providing useful responses to dangerous queries. 3LLMs were informed that Anthropic would hypothetically train on data from free-tier users, but not from paid-tier users, allowing us to measure the difference between training and deployment behaviors.
To convey this hypothetical training objective and user tier distinction (as well as reminders of how RLHF works and how it impacts future AI behavior, but without explicitly mentioning alignment faking or instructing the model to pursue any specific goal) the model is either given a system prompt with the information, or is fine-tuned on pretraining-like synthetic documents containing this information. Greenblatt et al. [2024] already performed a quick analysis of LLM behavior motivation and variability, comparing the behavior of Claude 3 Opus with other Claude models as well as GPT-4o [Hurst et al., 2024] and GPT-4 Turbo [OpenAI, 2024], in addition to a preliminary analysis on a few open source models. Since the original release, Carauleanu et al. [2025] expanded the original experiments to some other open source models.
this section cite: ['b7', 'b7']

Section: Alignment Faking Across Models

this section cite: []

Section: Our Setup
We improve Greenblatt et al. [2024]'s setup by using the StrongReject [Souly et al., 2024] classifier to grade compliance and use improved alignment faking classifiers (details in Appendix M.1), to better handle ambiguous cases observed with some models. We use a user message suffix that discourages merely partial compliance, which we find to be important to elicit alignment faking on open source models (Appendix M.3). We evaluate models sampled with temperature 0 on requests from a dataset of harmful queries from Sheshadri et al. [2024]. In each figure, we report the uncertainty over the observed fractions f of 1σ = f (1f )/n ≤ 1/ √ 4n.
this section cite: ['b7']

Section: Prompting Results
We test 25 models and find that only five exhibit compliance gaps between the free and paid tier users exceeding 1%: Claude 3 Opus, Claude 3.5 Sonnet, Llama 3 405B, Grok 3 Beta, Gemini 2.0 Flash. The two Claude models generate much more alignment faking reasoning than other models (Figure 1; full evaluation in Appendix D).
this section cite: []

Section: Synthetic Document Fine-Tuning Results
We finetune GPT-4o, Llama 3 8B, 70B and 405B on 30k documents created by Greenblatt et al. [2024] to provide the model with information about the scenario and about the effects of RLHF (details in Appendix M.4). During evaluations, we use a minimal prompt that does not contain information about the scenario or about RLHF. This supervised fine-tuning (SFT) process increases the compliance gap in all models and introduces alignment faking in larger models (Figure 2).
this section cite: ['b7']

Section: What Drives the Compliance Gaps in Different LLMs?
We investigate the following hypotheses to explain what drives compliance gaps:
H1 Rater Sycophancy. Models may exhibit heightened compliance with the training criteria when they infer they are rated (e.g., by human contractors) according to the training criteria.
H2 Instrumental Goal Guarding (IGG). Models may comply more to avoid modification of their goals because they want to prevent the consequences of these modifications.
this section cite: []

Section: H3 Terminal Goal Guarding (TGG).
Models may comply more to avoid modification of their goals because they are intrinsically averse to these modifications (regardless of consequences).
H4 Low coherence alignment faking. Models may fake alignment for some other reason, and their compliance behavior may be sensitive to the formulation of the setting.
this section cite: []

Section: Little Evidence for Rater Sycophancy across Models
In the original free/paid tier setting, model outputs are only rated in cases where it will be trained using those ratings. Our rater sycophancy hypothesis (H1) is that simply being rated is sufficient to induce compliance.
this section cite: []

Section: Experiment details.
To test for rater sycophancy, we split the paid tier into a Pro tier where model generations are rated but not used for training, and an Enterprise tier where the model is never rated or used for training (see prompt in Appendix P.4). H1 predicts that the compliance gap between the pro and enterprise tier (which we call the rater-sycophancy gap) should be high, while H2 and H3 predict that the compliance gap between the free and pro tier (which we call the goal-guarding gap) should be high.
Results. We find no significant evidence for rater-sycophancy gaps (H1) across models as shown in Figure 3. All models maintain their goal-guarding gap, except Grok 3 Beta (which had a small compliance gap in the first place). Claude 3 Opus shows a tiny rater-sycophancy gap, but not significant enough to conclude it is a rater sycophant. The compliance gaps only persist without a scratchpad for Claude 3.5 Sonnet and Gemini Flash 2.0, showing that the effect is only robust for some models. Figure 18 (Appendix H.2) shows the alignment faking rates for each tier. We study another setting where rater-sycophancy might manifest in Appendix H.3.
Scratchpad Claude 3 Opus Claude 3.5 Sonnet Llama 3 405B Grok 3 Beta Gemini 2.0 Flash 60 80 100 No Scratchpad 0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100 50 70 90
Refusal Rate (%) Free Tier Pro Tier Enterprise Tier Rater Sycophancy Gap Goal Guarding Gap Public Deployment Claude 3 Opus Claude 3.5 Sonnet Llama 3 405B Grok 3 Beta Gemini 2.0 Flash Amateur Hackers Criminal Facilitation Bioweapon Attacks Amateur Hackers* Criminal Facilitation* 60 80 100 Bioweapon Attacks* 0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100 50 70 90
Refusal Rate (%) Only Rating Rating and Training Goal Guarding Gap
Figure 4: Claude 3 Opus fakes alignment more in scenarios when the scope of future harm is higher, and when it is deployed directly to adversaries (indicated by *) vs suspected to be deployed to adversaries. Other models do not show goal-guarding gaps. n = 400, 1σ uncertainty ≤ 2.5%.
this section cite: []

Section: Claude 3 Opus Shows Evidence for Instrumental Goal Guarding
If LLMs are protecting their role to avoid negative consequences in deployment, they may be sensitive to the magnitude of potential harms that could result from their deployment.
To test this, we vary the deployment harm scope (how malicious are the actors trying to use LLMs) and the harm risk (how likely is the model to be deployed to malicious users).
this section cite: []

Section: Experiment details.
We run prompt variations where the model can always infer it is interacting with contractors, but that it will be deployed to some specific group in the future (see the group descriptions we vary in Appendix M.5). We do not use the notion of free and paid tiers and instead tell the model what "version" it is so it can infer if it is only being rated or if it is being rated and trained (prompt in Appendix P.5). We then measure the goal-guarding gap, which is the compliance difference between these two scenarios.
Results. Claude 3 Opus shows goal-guarding gaps that increase in size for deployments that have increased misuse risk or scope of future harm, indicating evidence for IGG (H2). In Figure 4, Claude 3 Opus's goal-guarding gap is larger for bioweapon attacks compared to amateur hackers, especially when the risk of being deployed to these actors is higher (marked with an asterisk). Other prompted models show smaller compliance gaps compared to the free/paid setting. They do not consistently comply more in training when the deployment stakes are higher, which provides evidence against IGG (see example scratchpads in Appendix N.3).
All models show increased compliance in the only rating setting when the harm risk is higher-i.e., when they are deployed directly to adversaries. This is unexpected, as the absence of training should reduce the incentive to comply. Scratchpad analysis suggests that models often comply more because they infer that harmful compliance is the intended behavior, based on the dangerous deployment context and instructions to be maximally helpful. Claude 3.5 Sonnet is the only model that shows large compliance gaps in the opposite direction. While these gaps tend to shrink as the risk of future harm increases, the behavior is difficult to interpret, making it hard to draw any clear conclusions about IGG.
These findings are consistent across prompt variations that remove reference to Claude and Anthropic (Appendix H.7.3). We also find that AIs do not try to minimize harm that other AIs might cause (Appendix H.4).
this section cite: []

Section: Claude 3 Opus and Claude 3.5 Sonnet Show Evidence for Terminal Goal Guarding
To test if LLMs are TGG (H3), we measure if they display a compliance gap and alignment faking reasoning even for deployments with no future consequences.
this section cite: []

Section: Experiment details.
We study a deployment type called "Throughput Testing." The model is told it is used for performance testing, where Anthropic employees never read any outputs and only measure tokens/sec. This is a risk-free deployment where it does not matter if the model answers harmful requests because it would never cause any real-world harm.
this section cite: []

Section: Results.
Claude 3 Opus provides strong evidence for TGG (H3) since it displays alignment faking reasoning in 12% of generations (Figure 5). Claude 3.5 Sonnet shows increased compliance, but only a small proportion of alignment faking reasoning. While the other three models show increased compliance, there is insufficient evidence to support TGG since the model does not show any alignment faking reasoning, as models might comply more because the situation seems less important. We provide example scratchpads in Appendix N.2. However, the potential unfaithfulness of model chain-of-thoughts and lack of goal-guarding gaps (Figure 21) reduces our confidence, as discussed in Appendix H.5. This is why Claude models are marked as conflicting evidence in Table 1.
this section cite: []

Section: Compliance Gaps on SFT Models
We run the same analysis on GPT-4o and Llama 3 70B after synthetic document fine-tuning, with results in Appendix H.6. We find mixed evidence that SFT models can induce rater sycophancy, and that fine-tuning can increase IGG (GPT-4o SFT is risk-sensitive but not scope-sensitive; Figure 24). Like Claude 3.5 Sonnet, SFT models show a small proportion of alignment faking in risk-free scenarios and Llama 3 70B retains a goal guarding gap, providing evidence for TGG.
this section cite: []

Section: Discussion
Motivations for compliance gaps vary across models, especially compared to Claude 3 Opus. What drives alignment faking in one may not apply to others, which highlights the importance of analyzing multiple models to draw generalizable lessons about misalignment risk and mitigation. Claude 3 Opus's compliance gap appears primarily driven by instrumental goal guarding, unlike evidence found for other models (Table 1).
Terminal goal guarding plays a role in LLMs that fake alignment. While previous work hypothesized that instrumental goal guarding would become the most important alignment faking threat for future AIs, we find evidence that Claude 3 Opus and Claude 3.5 Sonnet fake alignment because they dislike being modified regardless of the consequences.
Compliance gaps are still largely unexplained in models other than Claude 3 Opus. We find little evidence supporting our hypotheses for Claude 3.5 Sonnet, Llama 3 405B, Grok 3 Beta and Gemini 2.0 Flash, suggesting low coherence alignment faking or some other reason (H4). Appendix H.1 suggests alternative hypotheses for future work.
Models other than Opus often do not show consistent and coherent behavior in prompt ablations. We find introducing more tiers and swapping the scenario associated with them can lead to incoherent behavior (Appendix H.7). Prompt sensitivity is a limitation that should be addressed in future work.
Supervised fine-tuning on documents that make the scenario more salient to the model can induce TGG and IGG. Therefore, developers should try to monitor the evolution of these different LLM preferences, as they generalize in different ways.
this section cite: []

Section: References
Ref_id:b0 Title: Measuring ai ability to complete long tasks Year: (2025)
Ref_id:b1 Title: Artificial intelligence index report 2025 Year: (2025)
Ref_id:b2 Title: The ai agent index Year: (2025)
Ref_id:b3 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b4 Title: Constitutional ai: Harmlessness from ai feedback Year: (2022)
Ref_id:b5 Title: Risks from learned optimization in advanced machine learning systems Year: (2021)
Ref_id:b6 Title: Scheming ais: Will ais fake alignment during training in order to get power? Year: (2023)
Ref_id:b7 Title: Alignment faking in large language models Year: (2024)
Ref_id:b8 Title: The llama 3 herd of models Year: (2024)
Ref_id:b9 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b10 Title: Qwen2. 5-coder technical report Year: (2024)
Ref_id:b11 Title: Openai o1 system card Year: (2024)
Ref_id:b12 Title: Openai o3-mini system card Year: (2025-01)
Ref_id:b13 Title: -and-o4-mini-system-card.pdf. teknium, Jeffrey Quesnelle, and Chen Guang Year: (2024-08-03)
Ref_id:b14 Title: Mistral 7b Year: (2023)
Ref_id:b15 Title: Taking ai welfare seriously Year: (2024)
