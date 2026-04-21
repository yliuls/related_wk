Title: Non-stationary Diffusion For Probabilistic Time Series Forecasting
Abstract: Due to the dynamics of underlying physics and external influences, the uncertainty of time series varies over time. However, existing Denoising Diffusion Probabilistic Models (DDPMs) fail to capture this non-stationary nature, constrained by their constant variance assumption from the additive noise model (ANM). In this paper, we innovatively utilize the Location-Scale Noise Model (LSNM) to relax the fixed uncertainty assumption of ANM. A diffusion-based probabilistic forecasting framework, termed Nonstationary Diffusion (NsDiff), is designed based on LSNM that is capable of modeling the changing pattern of uncertainty. Specifically, NsDiff combines a denoising diffusion-based conditional generative model with a conditional mean and a variance estimator, enabling adaptive endpoint distribution modeling. Furthermore, we propose an uncertainty-aware noise schedule, which dynamically adjusts the noise levels to accurately reflect the data uncertainty at each step and integrates the time-varying variances into the diffusion process. Extensive experiments conducted on nine real-world and synthetic datasets demonstrate the superior performance of NsDiff compared to existing approaches. Code is available at https:  //github.com/wwy155/NsDiff.

Section: Introduction
Time series forecasting plays a key role in various fields such as traffic prediction (Ermagun & Levinson, 2018) and supply chain management (Chopra & Meindl, 2021). Given a historical multivariate series X, general forecasting methods involve training an f (X) to predict a future series Y, which can be viewed as modeling the E [Y|X]. Although recent research has demonstrated promising capabilities to model conditional expectations (Zhou et al., 2021;Wu et al., 2020), effective decision-making, particularly in high-stakes fields like healthcare (Bertozzi et al., 2020) and finance (Li & Bastos, 2020), often requires accurately estimating the uncertainty underlying the data (Kendall & Gal, 2017;Xu et al., 2024). To address this problem, many recent studies have focused on probabilistic time series forecasting (Rasul et al., 2021;Chen et al., 2024;Li et al., 2024b), where the goal to estimate a distribution of possible future outcomes along with their associated uncertainties. The Denoising Diffusion Probabilistic Models (DDPMs) have recently gained significant attention for probabilistic time series forecasting due to their powerful ability to generate high-dimensional data (Rasul et al., 2021;Tashiro et al., 2021;Li et al., 2024a). Existing DDPMs typically rely on the Additive Noise Model (ANM) (Spirtes et al., 2001), which assumes Y = f (X) + ϵ, where ϵ ∼ N (0, σ) represents stationary Gaussian noise. The primary objective of these models is not only to estimate the conditional expectation E[Y|X] via f (X), but also to accurately capture uncertainty by modeling the noise distribution ϵ. While DDPMs with stationary Gaussian noise have achieved sub-stantial success in domains such as computer vision and natural language generation (Ho et al., 2020;Dhariwal & Nichol, 2021;Gu et al., 2022), they are less effective for modeling non-stationary time series data, where patterns of uncertainty vary contextually (Lee et al., 2024). Figure 1 illustrates an example from the ILI (influenzalike illness) dataset, with different endpoint distributions (Left) and estimated uncertainty (Right) on different models: TimeGrad (Rasul et al., 2021), TMDM (Li et al., 2024b), and NsDiff (ours). In the upper part of Figure 1, TimeGrad (Rasul et al., 2021) employs the endpoint N (0, I), which fails to capture non-stationary characteristics. TMDM (Li et al., 2024a) uses N (f (X), I) as endpoint, representing changing averages. On the test dataset (shown to the right of the red dashed line), where both the number of patients and the corresponding deviation increase, the performance differences are evident. TimeGrad fails to model both the underlying trends and deviations. In contrast, TMDM effectively captures the trends through its f (X), but its stationary covariance I limits its ability to accurately estimate uncertainty, which is critical for the probabilistic time series forecasting.
To better address non-stationarity with changing uncertainty, we introduce Location-Scale Noise Model (LSNM) into DDPMs, which relaxes the traditional Additive Noise Model (ANM) by incorporating a contextually changing variance: Y = f (X) + g(X)ϵ, where g(X) is an X-dependent variance model and ϵ is a standard gaussian noise. LSNM is capable of modeling both the contextual mean through f (X) and the contextual uncertainty through g(X). In the special case where g(X) ≡ 1, this simplifies to the standard ANM. Building upon this more flexible and expressive assumption, we propose the Non-stationary Diffusion Model (NsDiff) framework, which provides an uncertaintyaware noise schedule for diffusion process. In summary, our contributions are:
• We observe that the ANM is inadequate for capturing the varying uncertainty and propose a novel framework that integrates LSNM to allow for explict uncertainty modeling. This work is the first attempt to introduce LSNM into probabilistic time series forecasting.
• To fundamentally elevate the noise modeling capabilities of DDPM, we seamlessly integrate time-varying variances into the core diffusion process through an uncertainty-aware noise schedule that dynamically adapts the noise variance at each step.
• Experimental results indicate that NsDiff achieves superior performance in capturing uncertainty. Specifically, in comparison to the second-best recent baseline TMDM, NsDiff improves up to 66.3% on real-world datasets and 88.3% on synthetic datasets.
this section cite: ['b7', 'b4', 'b47', 'b41', 'b1', 'b21', 'b15', 'b42', 'b33', 'b3', 'b33', 'b37', 'b36', 'b13', 'b5', 'b10', 'b20', 'b33', 'b33']

Section: Related Works

this section cite: []

Section: DDPM for Probabilistic Forecasting
Denoising Diffusion Probabilistic Models (DDPMs) have shown promising results in the probabilistic forecasting area (Tyralis & Papacharalampous, 2022). Rasul et al. (2021) introduce TimeGrad, an autoregressive diffusion model guided by a recurrent neural network hidden state. Tashiro et al. (2021) propose a masking strategy for training diffusion models, applicable to tasks like imputation and forecasting. Alcaraz & Strodthoff (2022) extend DDPMs with a structured space model to capture long-term dependencies. TimeDiff (Shen & Kwok, 2023) utilized future mixup and autoregressive initialization. Li et al. (2022b) integrate multiscale denoising score matching to guide the diffusion process, ensuring generated series align with the target. DiffusionTS (Yuan & Qiao, 2024) trains the model to reconstruct the sample rather than noise, using a Fourierbased loss term.
Kollovieh et al. (2024) propose a selfguiding strategy for time series generation and forecasting based on structured state-space models. By leveraging the bridge-based model introduced by Shi et al. (2023), Chen et al. (2023) present a convergence analysis of the Schrödinger bridge algorithm and propose improvements to the diffusion process. However, these methods generally assume fixed endpoint variance, which is hard to model non-stationary time series.
this section cite: ['b38', 'b33', 'b37', 'b0', 'b34', 'b44']

Section: Non-stationary Time Series Forecasting
To address non-stationarity,
Li et al. (2022a) employ a domain-adaptation approach to predict data distributions, while Du et al. (2021) propose an adaptive RNN for distribution matching to mitigate non-stationary effects. Liu et al. (2022) introduce a non-stationary Transformer with destationary attention to account for non-stationary factors in self-attention. Wang et al. (2022) use global and local Koopman operators to capture patterns at different scales, and Liu et al. (2024a) apply Koopman operators to components identified via Fourier transforms. Other approaches decompose stationary and non-stationary parts, such as Ogasawara et al. (2010) with local normalization, and Passalis et al. (2019)
with a learnable, instance-wise normalization. RevIN (Kim et al., 2021) addresses the distribution shift using reversible normalization, and recent works (Fan et al., 2023;Liu et al., 2024b) explore finer-grained trend modeling. Jiang et al. (2023) addresses non-stationarity in chaotic systems by preserving invariant measures to stabilize dynamical systems over time without relying on domain-specific priors. Fourier transforms, closely linked with non-stationarity, are also been applied to tackle these issues (Fan et al., 2024;Ye et al., 2024). Despite these advances in time series forecasting, the non-stationary uncertainty in probabilistic forecasting remains largely unexplored.
this section cite: ['b16', 'b8', 'b14', 'b9', 'b43']

Section: Preliminary

this section cite: []

Section: Problem Formulation
Given a historical multivariate time series X ∈ R N ×D where N is the historical window size and D denotes the number of feature dimensions. The probabilistic forecasting task is to predict the distribution of the future multivariate time series Y = {p(y 1 ), p(y 2 ), ..., p(y M )|y ∈ R D }, where M is the future window size. While previous works model the future series with ANM: Y = f ϕ (X) + ϵ, we model it based on LSNM with a more generalized data model:
Y = f ϕ (X) + g ψ (X)ϵ(1)
where the ϵ ∼ N (0, σ) is Gaussian noise. The f ϕ (X) and g ψ (X) can be viewed as prior knowledge with pre-trained parameters ϕ and ψ, where the f ϕ (X) is modeling the conditional expectation E[Y|X] and g ψ (X) is modeling the varying uncertainty. In this paper, we incorporate this two prior knowledge into the diffusion model to tackle the nonstationary challenge in probabilistic time series forecasting.
this section cite: []

Section: Denoising Diffusion Probabilistic Models
DDPMs (Ho et al., 2020) is a popular generative model to estimate the uncertainty for future time series. In the original DDPM, the future series distribution can be represented as p θ (Y 0 ) := p θ (Y 0:T )dY 1:T , where Y 1 , ..., Y T are latent variables. The joint distribution is defined as a Markov chain p θ (Y 0:T ) := p(Y T ) T t=1 p θ (Y t-1 |Y t ), where the endpoint of diffusion is set to p(Y T ) := N (0, I). To generate the distribution p θ (Y 0 ), DDPM designs two processes: a forward process to gradually add noise and a reverse process to denoise. In the forward process, the future series Y 0 gradually diffuses to the given prior endpoint Y T without any trainable parameters.
q(Y 1:T |Y 0 ) := T t=1 q(Y t |Y t-1 ) q(Y t |Y t-1 ) := N (Y t ; 1 -β t Y t-1 , β t I)(2)
where β t ∈ (0, 1) is a diffusion schedule for controlling the endpoint Y T ∼ N (0, I). This forward sampling can be simplified by q(Y t |Y 0 ) = N (Y t ; √ ᾱt Y 0 , (1 -ᾱt )I) in practice, where α t := 1 -β t and ᾱt := t i=1 α i . The reverse process parameterizes p θ (Y t-1 |Y t ) and compares it against forward process posteriors q(Y t-1 |Y t , Y 0 ). DDPM has shown that matching these two posteriors is equivalent to estimating the added noise η in the forward process. Thus, the parameterization of p θ (Y t-1 |Y t ) is:
p θ (Y t-1 |Y t ) := N (Y t-1 ; µ θ (Y t , t), 1 -ᾱt-1 1 -ᾱt β t I) µ θ (Y t , t) := 1 √ α t (Y t - β t √ 1 -ᾱt η θ ))(3)
where η θ is the estimated noise by a denoising model, which optimizes the following objective:
E Y0∼q(Y0),η∼N (0,I),t ||η -η θ || 2(4)
Following this basic forward and reverse process, many diffusion-based methods improve the reverse process (Rasul et al., 2021;Shen & Kwok, 2023) or prior distribution (Li et al., 2024a) with the historical time series information. However, they fix the variance of the prior distribution and focus on the expectation matching. The prior setup and training of uncertainty are largely ignored.
this section cite: ['b13', 'b33', 'b34']

Section: Methodology
In this section, we introduce the proposed NsDiff, including the design of forward and reverse process distributions, as well as the training and inference procedures of NsDiff. Furthermore, we discuss two simplified versions of NsDiff.
The outline of NsDiff is given in Figure 2.
this section cite: []

Section: Forward and Reverse Process
In previous diffusion-based methods, the uncertainty prior was missing, and they tended to set the endpoint of diffusion Y T to N (0, I) or N (f ϕ (X), I). To improve this, we use a different noise model LSNM to form the endpoint:
p(Y T |f ϕ (x), g ψ (x)) := N (f ϕ (X), g ψ (X))(5)
where f ϕ (X) models the conditional expectation E[Y|X] which can be parameterized by any forecasting model, e.g., Dlinear (Zeng et al., 2023) or PatchTST (Nie et al., 2022). We follow previous works (Kim et al., 2021;Liu et al., 2024b) to train the prior scale of uncertainty g ψ (X). We use the input variance to predict the output variance.
The forward process incrementally modifies the noise at each step to approach the endpoint distribution. To seamlessly integrate time-varying variances into the diffusion process, we propose an uncertainty-aware noise schedule, and incorporate data variance into the forward process dis-
tribution q(Y t | Y t-1 , f ϕ (X), g ψ (X), σ Y0 ).
Specifically, given well-pretrained models f ϕ , g ψ , and a prior state Y t-1 , we control the scaled variance to transition from the actual variance σ Y0 at the starting point to the endpoint g ψ (X).
The resulting distribution is normally distributed as:
N (Y t ; √ α t Y t-1 + (1 - √ α t )f ϕ (X), (β 2 t g ψ (X) + α t β t σ Y0 ) σt )(6)
where the shared coefficient β t is a noise scaling constant. As the noise step t increases, the term β t g ψ (X) grows and α t σ Y0 decreases. At t = T , where α t = 0, the variance converges to the assumed endpoint g ϕ (X). This enables DDPM to adaptively adjust the noise levels at each step to capture the data uncertainty. The forward distribution admits a closed-form sampling distribution q(Y t |Y 0 , f ϕ (X), g ψ (X), σ Y0 ) with an arbitrary timestep t:
N (Y t ; √ ᾱt Y 0 + (1 - √ ᾱt )f ϕ (X), ( βt -βt )g ψ (X) + βt σ Y0 σt )(7)
where we define the following coefficients:
αt := t-1 k=0 t i=t-k α i , βt := 1 -ᾱt αt := t-1 k=0 t i=t-k α i α t-k , βt := αt -αt .(8)
we leave the detailed derivation of σt to Appendix A.1, and all these coefficients are positive numbers. Notably, under a perfect estimator (assuming g ψ (X) = σ Y0 ), σt simplifies to βt g ϕ (X), and with the additional assumption of σ Y0 = I, it degenerates to the earlier constant variance settings ( βt I).
More detailed discussions and derivations can be found in Section 4.6 and Appendix A.5.
In the reverse process, the posteriors of Y t-1 are tractable when conditioned on Y 0 , which can be restated as:
q(Y t-1 |Y t , Y 0 , f ϕ (X), g ψ (X), σ Y0 ) := N (Y t-1 ; μ, σ)(9) where
μ := γ 0 Y 0 + γ 1 Y t + γ 2 f ϕ (X)(10)
σ := σ t σ t-1 α t σ t-1 + σ t(11)
and γ 0,1,2 in μ are given as:
γ 0 := √ ᾱt-1 σ t α t σt-1 + σ t , γ 1 := √ α t σt-1 α t σt-1 + σ t γ 2 := √ α t (α t -1) σt-1 + (1 - √ ᾱt-1 )σ t α t σt-1 + σ t (12
)
We leave the derivation in Appendix A.2. We follow the basic step of DDPM to parameterize a denoise model
p θ (Y t-1 |Y t , f ϕ (X), g ψ (X)) to match the forward process posteriors q(Y t-1 |Y t , f ϕ (X), g ψ (X), σ Y0 ).
this section cite: ['b45', 'b30', 'b16']

Section: Loss Function
We approximate the denoising transition step
p θ (Y t-1 |Y t , f ϕ (X), g ψ (X)) to the ground-truth de- noising transition step q(Y t-1 |Y t , f ϕ (X), g ψ (X), σ Y0 )
by optimizing the KL divergence (Hershey & Olsen, 2007) between the posterior distribution q and the parametrized distribution p θ . Like classic DDPM, we optimize only the diagonal variance term, denoted as σ and σ θ respectively. The loss is defined as the KL divergence of the noise matching term:
L = E [D KL (N x; μ, σ∥N (y; µ θ , σ θ ))] ∝ E ||η -η θ || 2 2 + i σi σ θ,i - i log σi σ θ,i(13)
where η θ is the estimated noise and η is the ground truth noise. The first term ensures the estimation of the posterior mean, while the rest terms guarantee the estimation of the variance. We provide the proof in Appendix A.3.
this section cite: ['b12']

Section: Pretraining f ϕ and g ψ
To train f ψ , we follow prior work (Li et al., 2024a) and utilize the Non-stationary Transformer (Liu et al., 2022) as the backbone model. The training process is identical to that of standard supervised time series models (Zhou et al., 2021). For the training of g ψ (X), we use a sliding window approach to extract the estimated ground truth variance, similar to references (Kim et al., 2021;Liu et al., 2024b;Ye et al., 2024). Specifically, given time series label Y 0 the estimated ground truth variance is defined as:
σ Y0 = Var(SlidingWindow(Y 0 )) (14
)
thus, the training of g ψ (X) is formulated as a supervised task. In our implementation, we utilize a sliding stride of 1 and a window size of 96. The function g ψ is implemented as a three-layer MLP, with outputs passed through the softplus activation (Zheng et al., 2015) to ensure positivity. Further implementation details can be found in Appendix C.2 and we examine the necessity of pretraining in Appendix B.2.
this section cite: ['b26', 'b47', 'b16', 'b43', 'b46']

Section: Training NsDiff
The target of NsDiff training is to match posterior distribution q by parameterizing p θ . Like traditional DDPM, NsDiff can be trained end-to-end by sampling a random t and noise η from uniform and Gaussian distributions respectively. According to Eq. 13, we build an estimation model ξ θ (Y t , f ϕ (X), g ψ (X), t) during the training process to match the noise and variance. The overall procedure is presented in Algorithm 1.
this section cite: []

Section: Inference
The target of the inference phase is to recursively sample from the parameterized distribution
p θ (Y t-1 | Y t , f ϕ (X), g ψ (X)), we provide a detailed
Algorithm 1 Training Input: Data X, target Y, model f ϕ , noise and variance estimation model ξ θ , total timesteps
T Pre-train f ϕ (X) to predict E(Y|X) Pre-train g ψ (X) to predict Var(Y|X) repeat Draw Y 0 ∼ q(Y 0 | X) Draw t ∼ Uniform({1, . . . , T }) Draw η ∼ N (0, I) Compute Y t : Y t = √ ᾱt Y 0 + (1 - √ ᾱt )f ϕ (X) + ( βt -βt )g ψ (X) + βt σ Y0 η ▷ using Eq. 7 Compute estimated noise and variance: η θ , σ θ = ξ θ (Y t , f ϕ (X), g ψ (X), t) Compute loss L ▷ using Eq. 13 Numerical optimization step on ∇ θ L until Convergence
process in Algorithm 2. At the inference phase, according to Eq. 10 and 11, the calculation of the parameters for the reverse distribution requires estimating both Y 0 and σ Y0 . For the estimation of Y 0 , we follow prior work (Han et al., 2022) and utilize the relationship between Y t and Y 0 as defined in Eq. 7. However, the estimation of σ Y0 lacks a direct correspondence with Y t . To estimate σ Y0 , one straightforward approach is to directly use g ψ (X). However, it demands a perfect predictor and does not incorporate the reverse process into parameter estimation. Actually, Eq. 11 can be expanded as a quadratic equation with respect to σ Y0 . Thus, we utilize the quadratic expansion of Eq. 11 to approximate σ Y0 , we leave the detailed derivation at Appendix A.4. Specifically, expanding Eq. 11 gives the following solvable equation:
λ 0 σ 2 Y0 + λ 1 σ Y0 + λ 2 = 0(15)
where the coefficients are
λ 0 := α t β t βt-1 λ 1 := β 2 t βt-1 + α t β t ( βt-1 -βt-1 )g ψ (X)- σ θ (α t βt-1 + α t β t )) λ 2 := g ψ (X) 2 β 2 t ( βt-1 -βt-1 )- σ θ g ψ (X)(α t βt-1 -α t βt-1 + β 2 t ) (16
)
λ 0 is a positive value, and according to Vieta's theorem (Lang, 2012), when λ 2 < 0, the equation has exactly one positive root. The constraint for λ 2 < 0 is equivalent to:
g ψ (X) < σ θ α t β 2 t + 1 βt-1 -βt-1 (17
)
Therefore, the solvability of the equation is governed by the noise level parameter β t . Under the typical DDPM parameterization (Ho et al., 2020), where β t ranges from 0.0001
Algorithm 2 Inference Input: data X, models f ϕ , g ψ , and
ξ θ Initialize Y T ∼ N (f ϕ (X), g ψ (X)) for t = T to 1 do if t > 1 then Draw z ∼ N (0, I) end if Compute η θ , σ θ = ξ θ (Y t , f ϕ (X), g ψ (X), t) Compute σY0 = -λ1+ √ λ 2 1 -4λ0λ2 2λ0 ▷ using Eq. 18 Compute Ŷ0 = 1 √ ᾱt Y t -(1 - √ ᾱt ) f ϕ (X) - ( βt -βt )g ψ (X) + βt σY0 η θ ▷ using Eq. 7 if t > 1 then Set Y t-1 = γ 0 Ŷ0 + γ 1 Y t + γ 2 f ϕ (X) + √ σ θ z else Set Y t-1 = Ŷ0 end if end for
Output: Y 0 to 0.02, the coefficient on the right-hand side of Eq. 17 becomes sufficiently large, thereby ensuring the equation's solvability. Hence, by solving the quadratic equation in Eq. 15, we can estimate the value of σ Y0 during inference stage, the specific formula is given by Eq. 18:
σY0 = -λ 1 + λ 2 1 -4λ 0 λ 2 2λ 0 (18
)
Experimentally, the approach exhibits consistent solvability across all datasets. We provide more discussions in Appendix A.4.
this section cite: ['b11', 'b19', 'b13']

Section: Simplified Variants of NsDiff
In this section, we discuss two simplified versions of NsDiff by simplifying the variance terms in Eq. 7. We summarize these two variants in Table 1, and provide the ablation results in Section 5.3.
w/o LSNM N f ϕ (X), I) βtI w/o UANS N (f ϕ (X), g ψ (X)) βtg ψ (X) NsDiff N (f ϕ (X), g ψ (X)) β 2 t g ψ (X) + βtαtσ Y 0
Perfect Estimator (w/o UANS): Assuming a perfect variance estimator g ψ (X) = σ Y0 , Eq. 6 becomes the following:
N (Y t ; √ α t Y t-1 + (1 - √ α t )f ϕ (X), (1 -α t ) g ψ (X))(19)
Further derivations show that this is simply a constant multiplication of the variance term from prior works (Han et al., 2022), and the training of the variance is not necessary. However, this estimation of uncertainty has two main drawbacks. First, assuming a perfect estimator inherently introduces bias. In addition, this approach estimates the variance without leveraging the denoising process, as the variance is fully determined by pretrained g ψ (X).
Unit Variance (w/o LSNM): Assuming a known unit variance, i.e., g ψ (X) = σ Y0 = I, Eq. 6 becomes:
N (Y t ; √ α t Y t-1 + (1 - √ α t )f ϕ (X), (1 -α t ) I). (20
)
which is consistent with previous work (Han et al., 2022). TMDM (Li et al., 2024a) is a typical probabilistic forecasting model built under this assumption. The results for TMDM are presented in Section 5.2 and 5.3, where we conduct experiments on real and synthetic datasets, respectively. We provide detailed derivations and more discussions in Appendix A.5.
this section cite: ['b11', 'b11']

Section: Experiments

this section cite: []

Section: Experiment Setup
Datasets: Nine popular real-world datasets with diverse characteristics are selected, including Electricity (ECL), ILI, ETT{h1, h2, m1, m2}, ExchangeRage (EXG), Traffic, and SolarEnergy (Solar). Table 2 summarizes basic statistics for these datasets. To estimate uncertainty variation between the train and test datasets, we use the ratio of test variance to train variance, selecting the highest value across dimensions to capture non-stationary uncertainty. A detailed notebook on this calculation is available in our repository. For dataset splits, we follow previous time series prediction works (Wu et al., 2022;Li et al., 2024b): the ETT datasets are split 12/4/4 months for train/val/test, while others are split 7:1:2. Details can be found in Appendix C.1. & Kwok, 2023), TMDM (Li et al., 2024a) and Diffu-sionTS (Yuan & Qiao, 2024). Specifically, TMDM denoises from N (f ϕ (X), I) while others denoise from N (0, I).
Experiment Settings: Experiments are conducted under popular long-term multivariate forecasting settings, using an input length of 168 in all experiments. All experiments are run with seeds {1, 2, 3} for 10 epochs. We use the best result from the validation set to evaluate the model on the test set. The learning rate is set to 0.001, batch size of 32 and the number of timesteps T = 20, consistent with prior work (Rasul et al., 2021). We employ a linear noise schedule with β 1 = 10 -4 and β T = 0.02, in line with the setup used in conventional DDPM (Ho et al., 2020). At inference, we generate 100 samples to estimate the distribution. For the baseline models, we utilize their default parameters.
Metrics: Following prior work (Li et al., 2024a), we use two probabilistic forecasting metrics: Quantile Interval Coverage Error (QICE) (Han et al., 2022) and Continuous Ranked Probability Score (CRPS) (Matheson & Winkler, 1976). For both metrics, smaller values indicate better performance. Detailed formula is provided in Appendix C.3. We provide point forecast results at Appendix B.1.
this section cite: ['b40', 'b44', 'b33', 'b13', 'b11', 'b29']

Section: Main Experiments
To evaluate the performance of NsDiff in probabilistic multivariate time series forecasting, we tested it on nine realworld datasets and compared it to five competitive baselines.
The results, summarized in Table 3, show that NsDiff consistently achieves state-of-the-art (SOTA) performance, with superior uncertainty estimation capabilities, except on the Solar dataset, which exhibits low uncertainty variation (0.92 shown in Table 2). Compared to the second-best and previous SOTA TMDM, which uses an endpoint distribution of N (f ϕ (X), I), NsDiff demonstrates significant improvements, particularly in the uncertainty interval estimation metric (QICE). For example, QICE is reduced by 47.9% on ETTh1, 53.6% on ETTh2, 20.5% on ETTm1, and 66.3% on Traffic. Notably, on the Traffic dataset, which has the highest uncertainty variation (181.83), NsDiff achieves the most significant improvement, underscoring its strength in handling high-uncertainty scenarios.
this section cite: []

Section: Sample Showcases
To provide a clearer understanding of NsDiff's performance, we visualize a sample from the ETTh1 dataset in Figure 3. As shown, NsDiff effectively captures the uncertainty, even under the distribution shift between input and output. In contrast, TMDM, while capable of detecting mean variations, fails to adequately model the uncertainty due to its assumption of uncertainty invariance.
Other models, such as TimeGrad, CSDI, and TimeDiff, which begin denoising from N (0, I), struggle to capture both the mean and variance. For example, as seen on the right side of the figure, TimeGrad predicts a stable trend instead of the observed downward shift. This highlights the limitations of these models in handling non-stationary behavior. In contrast, NsDiff excels at modeling such nonstationary dynamics while providing accurate uncertainty estimation, demonstrating its robustness and effectiveness in challenging forecasting scenarios. We provide other showcases in Appendix D.
this section cite: []

Section: Experiments On Synthetic Data
To accurately evaluate NsDiff's performance under timevarying conditions, we designed two synthetic datasets using the LSNM. Specifically, the formula used is Y = m[t] + v[t]ϵ, where m and v defines the level of trend and uncertainty variation. In the linear setting, m increases linearly from 1 to 10, and v follows the same pattern. In contrast, for the quadratic setting, v grows quadratically from 1 to 100. The total length of the generated dataset is 7588, and we predict univariate features. The results of these experiments are summarized in Table 4. Further details about the dataset construction can be found in Appendix C.1.2. As shown in Table 4, NsDiff achieves remarkable performance under conditions with varying variance. Compared to the previous model, TMDM, in terms of QICE, NsDiff improves performance by 78.3% on the linear-growing variance dataset, and this improvement increases to 88.3% on the quadratic-growing variance dataset. These results demonstrate the superior performance of NsDiff in capturing uncertainty shifts.
this section cite: []

Section: Synthetic Dataset Showcases.
To visually illustrate whether NsDiff can capture the uncertainty shift between the training and test datasets, we provide an example of a linear synthetic dataset in Figure 4, where the estimations for training and extended testing samples are plotted. As shown in the figure, both TMDM and NsDiff effectively capture the uncertainty within the training set. However, in the testing area (to the right of the red dashed line), TMDM assumes invariant uncertainty, while NsDiff successfully captures the uncertainty shift. This clearly demonstrates that NsDiff effectively captures the distribution shift between the training and test datasets, whereas previous methods under ANM fail to do so.
this section cite: []

Section: Ablation Experiments
This section compares two simplified variants of NsDiff discussed in Section 4.6, the ablation experiments are conducted on ETTh1 dataset. The abaltion variants are : (1) w/o LSNM: without LSNM assumption, which assumes conditional unit constant variance (σ Y0 = I) (2) w/o UANS: without uncertainty-aware noise schedule, which assumes a perfect noise estimator (σ Y0 = g ψ (X)). The results, presented in Table 5, show that NsDiff achieves the best performance, not only in overall metrics but also in the stability of results (lower variance). Notably, while assuming a perfect uncertainty estimator (w/o UANS) improves CRPS by introducing variable uncertainty, it remains suboptimal in QICE compared to w/o LSNM and exhibits higher variance. This is likely due to potential overfitting of the variance estimator, as it fully relies on g ψ (X). These findings highlight the importance of a controllable noise schedule, rather than solely relying on a perfect g ψ (X).
this section cite: []

Section: Conclusion
In this paper, we present Non-stationary Diffusion (NsDiff), a novel class of conditional Denoising Diffusion Probabilistic Models (DDPMs) specifically designed to advance probabilistic forecasting. NsDiff represents the first attempt to integrate the Location-Scale Noise Model (LSNM) into probabilistic forecasting, providing a more flexible and expressive framework for uncertainty representation in the data. We introduce an uncertainty-aware noise schedule, which enhances the noise modeling capabilities of DDPMs by incorporating time-varying variances directly into the diffusion process. NsDiff provides a generalized framework that extends the flexibility of existing models; by incorporating a pretrained mean and variance estimator along with the designed noise schedule, NsDiff enables accurate uncertainty estimation, thereby opening new opportunities for advancing research in probabilistic forecasting.
this section cite: []

Section: References
Ref_id:b0 Title: Diffusion-based time series imputation and forecasting with structured state space models Year: (2022)
Ref_id:b1 Title: The challenges of modeling and forecasting the spread of covid-19 Year: (2020)
Ref_id:b2 Title: Provably convergent schrödinger bridge with applications to probabilistic time series imputation Year: (2023)
Ref_id:b3 Title: Probabilistic forecasting with stochastic interpolants and f\ Year: (2024)
Ref_id:b4 Title: Supply Chain Management: Strategy, Planning, and Operation Year: (2021)
Ref_id:b5 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b6 Title: Adarnn: Adaptive learning and forecasting of time series Year: (2021)
Ref_id:b7 Title: Spatiotemporal traffic forecasting: review and proposed directions Year: (2018)
Ref_id:b8 Title: Dish-ts: a general paradigm for alleviating distribution shift in time series forecasting Year: (2023)
Ref_id:b9 Title: Deep frequency derivative learning for non-stationary time series forecasting Year: (2024)
Ref_id:b10 Title: Vector quantized diffusion model for text-to-image synthesis Year: (2022)
Ref_id:b11 Title: Classification and regression diffusion models Year: (2022)
Ref_id:b12 Title: Approximating the kullback leibler divergence between gaussian mixture models Year: (2007)
Ref_id:b13 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b14 Title: Training neural operators to preserve invariant measures of chaotic attractors Year: (2023)
Ref_id:b15 Title: What uncertainties do we need in bayesian deep learning for computer vision? Advances in neural information processing systems Year: (2017)
Ref_id:b16 Title: Reversible instance normalization for accurate time-series forecasting against distribution shift Year: (2021)
Ref_id:b17 Title: Predict, refine, synthesize: Self-guiding diffusion models for probabilistic time series forecasting Year: (2024)
Ref_id:b18 Title: Modeling long-and short-term temporal patterns with deep neural networks Year: (2018)
Ref_id:b19 Title:  Year: (2012)
Ref_id:b20 Title: Adaptive noise schedule for time series diffusion models Year: (2024)
Ref_id:b21 Title: Stock market forecasting using deep learning and technical analysis: a systematic review Year: (2020)
Ref_id:b22 Title: Ddg-da: Data distribution generation for predictable concept drift adaptation Year: (2022)
Ref_id:b23 Title: Generative time series forecasting with diffusion, denoise, and disentanglement Year: (2022)
Ref_id:b24 Title: Transformer-modulated diffusion models for probabilistic multivariate time series forecasting Year: (2024)
Ref_id:b25 Title: Transformer-modulated diffusion models for probabilistic multivariate time series forecasting Year: (2024)
Ref_id:b26 Title: Non-stationary transformers: Exploring the stationarity in time series forecasting Year: (2022)
Ref_id:b27 Title: Learning nonstationary time series dynamics with koopman predictors Year: (2024)
Ref_id:b28 Title: Adaptive normalization for non-stationary time series forecasting: A temporal slice perspective Year: (2024)
Ref_id:b29 Title: Scoring rules for continuous probability distributions Year: (1976)
Ref_id:b30 Title: A time series is worth 64 words: Long-term forecasting with transformers Year: (2022)
Ref_id:b31 Title: Adaptive normalization: A novel data normalization approach for nonstationary time series Year: (2010)
Ref_id:b32 Title: Deep adaptive input normalization for time series forecasting Year: (2019)
Ref_id:b33 Title: Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting Year: (2021)
Ref_id:b34 Title: Non-autoregressive conditional diffusion models for time series prediction Year: (2023)
Ref_id:b35 Title: Diffusion schrödinger bridge matching Year: (2023)
Ref_id:b36 Title: Causation, prediction, and search Year: (2001)
Ref_id:b37 Title: Conditional score-based diffusion models for probabilistic time series imputation Year: (2021)
Ref_id:b38 Title: A review of probabilistic forecasting and prediction with machine learning Year: (2022)
Ref_id:b39 Title: Koopman neural forecaster for time series with temporal distribution shifts Year: (2022)
Ref_id:b40 Title: Timesnet: Temporal 2d-variation modeling for general time series analysis Year: (2022)
Ref_id:b41 Title: Connecting the dots: Multivariate time series forecasting with graph neural networks Year: (2020)
Ref_id:b42 Title: Ordering-based causal discovery for linear and nonlinear relations Year: (2024)
Ref_id:b43 Title: Frequency adaptive normalization for non-stationary time series forecasting Year: (2024)
Ref_id:b44 Title: Diffusion-ts: Interpretable diffusion for general time series generation Year: (2024)
Ref_id:b45 Title: Are transformers effective for time series forecasting Year: (2023)
Ref_id:b46 Title: Improving deep neural networks using softplus units Year: (2015)
Ref_id:b47 Title: Informer: Beyond efficient transformer for long sequence time-series forecasting Year: (2021)
