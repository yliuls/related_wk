Title: LATENT FOURIER TRANSFORM
Abstract: We introduce the Latent Fourier Transform (LATENTFT), a framework that provides novel frequency-domain controls for generative music models. LATENTFT combines a diffusion autoencoder with a latent-space Fourier transform to separate musical patterns by timescale. By masking latents in the frequency domain during training, our method yields representations that can be manipulated coherently at inference. This allows us to generate musical variations and blends from reference examples while preserving characteristics at desired timescales, which are specified as frequencies in the latent space. LATENTFT parallels the role of the equalizer in music production: while traditional equalizers operates on audible frequencies to shape timbre, LATENTFT operates on latent-space frequencies to shape musical structure. Experiments and listening tests show that LATENTFT improves condition adherence and quality compared to baselines. We also present a technique for hearing frequencies in the latent space in isolation, and show different musical attributes reside in different regions of the latent spectrum. Our results show how frequency-domain control in latent space provides an intuitive, continuous frequency axis for conditioning and blending, advancing us toward more interpretable and interactive generative music models.

Section: INTRODUCTION
Modern audio generation models often operate in a coarse-to-fine manner, generating progressively finer representations of the output signal in a conditional chain. In diffusion models (Kong et al., 2020;Liu et al., 2023;Huang et al., 2023), higher noise levels provide coarser representations, while lower noise levels provide finer representations. In autoregressive models like AudioLM (Borsos et al., 2023a) and MusicLM (Agostinelli et al., 2023), an encoding stage represents the input signal as a hierarchy of coarse-to-fine tokens, and a generative model attempts to predict fine tokens from coarser ones. This is also the case for masked token models (Garcia et al., 2023), discrete diffusion (Yang et al., 2023), and next-scale prediction (Qiu et al., 2024).
Since the generative process involves conditioning on coarse representations, it is natural to generate new samples using the coarse representations of a reference example. This type of conditioning has been used for stroke-based image editing and image translation (Meng et al., 2021;Choi et al., 2021). However, conditioning on small-or mid-scale features is harder, since the representations used by the generative model rarely capture these features in isolation. For instance, discrete representations define fine tokens relative to coarse ones via residual vector quantization (RVQ) (Zeghidour et al., 2021;Kumar et al., 2023), preventing them from being interpreted independently.
Conditioning on arbitrary timescales from a reference example would be useful in music, which contains slow-moving patterns (like chord progressions) and fast-moving patterns (like trills). Patterns occurring at different timescales may be desirable starting points for generating musical variations, but are difficult to specify precisely using text. Existing reference-based controls (Villa-Renteria et al., 2025;García et al., 2025) target attributes like pitch, loudness, and instrumentation, which are distributed across multiple timescales. While these methods provide control over various semantic axes, none directly expose the 'timescale' axis.
To address this, we explore the use of the Fourier transform, which provides a decomposition of a signal into oscillations at different frequencies. High frequencies capture the most rapid variations in the signal ('small-scale' characteristics), while low frequencies capture slow variations in the signal ('large-scale' characteristics). This representation has two benefits:
• First, frequency components are orthogonal, meaning that changing the signal's representation at one frequency does not affect the signal's representation at other frequencies. Thus, the Fourier transform provides an inductive bias for separating information across timescales.
• Second, the frequency axis provides an intuitive, continuous axis for specifying timescales precisely. The user can select for patterns based on the timescales in Hz at which they occur, instead of relying on heuristic approaches for timescale specification.
Our approach merges the Fourier transform with deep representation learning: we use a diffusion autoencoder (Preechakul et al., 2022) to capture musical patterns, and a latent-space Fourier transform to separate them by scale. To achieve synergy between these two components, we propose a simple end-to-end training framework: an encoder transforms audio into a time series of latent vectors, which is randomly masked in the Fourier domain. Then, a decoder attempts to use this frequency-masked latent sequence to reconstruct the audio with a diffusion-based objective.
After training, we can encode user-selected music into a sequence of latent vectors. Then, we can apply a Fourier transform to this latent sequence, creating a latent spectrum. The latent spectrum maps different musical patterns to different frequencies in it, which we refer to as latent frequencies. These latent frequencies correspond to the timescales at which the musical patterns occur. The user can hear different parts of the latent spectrum in isolation, or generate variations while conditioning on patterns at desired timescales, which are specified as latent frequencies. Separation between timescales also allows us to blend two musical examples together, retaining features at user-selected timescales from each. In short, we introduce novel frequency-based controls for generative models.
To explain these controls and their effects, we draw parallels between our framework and the equalizer (EQ), an essential tool in audio signal processing. The equalizer manipulates the audible spectrum, or the frequencies in the audio waveform within the limits of human hearing (20 -20,000 Hz). This shapes sonic characteristics like "warmth," "brightness," "clarity," and "shine," which relate to different frequency ranges (Izhaki, 2017, pp. 223-232). The equalizer is particularly crucial for mixing multiple musical elements together coherently, by highlighting frequencies from each element and ensuring that elements do not "clash" over similar frequency ranges (Owsinski, 2017, pp. 14, 160-161). Since the equalizer operates on audio waveform frequencies, it is unable to change musical or structural patterns (like notes or chords). These are more complex than waveform oscillations, and unfold on temporal scales below 20 Hz, where such oscillations are inaudible. Still, these structural patterns are also vital to combining multiple musical elements together in a coherent way.
By operating on the latent spectrum instead of the audible spectrum, our framework provides a complement to the traditional equalizer that operates on musical patterns instead of sonic qualities. For instance, we can blend sounds together in musically coherent ways, while preserving patterns from each sound at user-specified latent frequencies. This is akin to the way traditional EQs are used to mix sounds together in musically pleasant ways, by choosing which audible frequencies of each sound to highlight. We dub our framework LATENTFT, and show several applications:
1. LATENTFT can generate musically coherent variations of a given song, while preserving patterns at desired timescales. These timescales are specified as a mask over the latent frequency spectrum. (Sec. 4.2).
2. LATENTFT can blend two songs, preserving patterns from each at desired timescales. These timescales are specified as masks over the latent frequency spectrum. (Sec. 4.3).
3. We can 'zoom-in' on parts of the latent spectrum, allowing us to hear musical patterns at desired timescales, which are specified as latent frequencies (Sec. 4.5).
4. We can interpret the latent spectrum of a song, and show where various musical characteristics like genre, tempo, and pitch reside on the latent spectrum (Sec. 4.6).
We demonstrate these applications through quantitative metrics (Table 1), listening tests (Sec. 4.4), and qualitative examples, which can be found on our website 1 .
this section cite: ['b47', 'b58', 'b38', 'b0', 'b24', 'b92', 'b71', 'b61', 'b14', 'b94', 'b49', 'b87', 'b25', 'b70']

Section: RELATED WORK
Audio Generation. Recent years have witnessed a great expansion in audio-domain generative models, which operate in a continuous domain or by generating discrete tokens. Diffusion models (Sohl-Dickstein et al., 2015;Ho et al., 2020;Song et al., 2020) generate samples by iteratively denoising pure Gaussian noise. Other approaches to audio generation rely on discrete audio codec tokens (Zeghidour et al., 2021;Kumar et al., 2023), which compress audio into a multi-layer sequence of tokens, with successive layers capturing increasingly fine details. Token generation can proceed in an autoregressive (Borsos et al., 2023a;Copet et al., 2023;Agostinelli et al., 2023) or nonautoregressive (Garcia et al., 2023;Borsos et al., 2023b) manner, but in both cases, coarse tokens typically condition the generation of finer ones. We propose Fourier-based representations that let us condition on features at arbitrary scales. We compare our method to conditioning on intermediate or fine tokens in our Masked Token Model baseline.
this section cite: ['b80', 'b36', 'b82', 'b94', 'b49', 'b16', 'b0', 'b24']

Section: Controls for Audio and Music Generation.
Current audio generation methods offer global controls like text (Forsgren & Martiros, 2022;Huang et al., 2023;Liu et al., 2023;Copet et al., 2023;Agostinelli et al., 2023;Chen et al., 2024;Schneider et al., 2024;Evans et al., 2025), or time-varying controls based on musical attributes like pitch and loudness curves (Wu et al., 2024;García et al., 2025) or stems (Parker et al., 2024;Villa-Renteria et al., 2025). Different time-varying signals allow for control along different semantic axes, but not along the 'timescale' axis. These works mostly condition on the entire control signal, not selected frequency components. The exception is Sketch2Sound (García et al., 2025), which optionally smooths the pitch or loudness-based control signal using median filtering. Still, this type of filtering is heuristic, applies only to preserving large-scale features, and operates on hand-extracted features instead of latent ones. Guidance (Levy et al., 2023) and initial noise optimization (Novack et al., 2024) have also been used to control music generation using differentiable objectives. We use guidance for our tasks in our Guidance baseline.
this section cite: ['b23', 'b38', 'b58', 'b16', 'b0', 'b12', 'b76', 'b21', 'b89', 'b25', 'b67', 'b87', 'b25', 'b55', 'b66']

Section: Image Editing Frameworks.
The coarse-to-fine paradigm lends itself to image editing frameworks that generate variations of input examples based on their low-frequency features. SDEdit (Meng et al., 2021) enables stroke-based image generation and editing by adding white noise to a given reference (which acts like a heuristic low-pass filter), and running the denoising process.
Similarly, Iterative Latent Variable Refinement (ILVR) (Choi et al., 2021) can generate variations of images while preserving large-scale structure. During the denoising process, ILVR continually replaces the low-frequency components of the noisy sample with the low-frequency components of a (noised) reference, enabling image translation and stroke-based editing. ILVR does not condition on high-frequency or mid-frequency components, but we attempt this in our ILVR baseline.
this section cite: ['b61', 'b14']

Section: Fourier-Based Deep Learning.
While we apply the Fourier transform to latent vectors, many works use frequency-domain representations of the input or output space. These include works in vision (Lee et al., 2018;Yang & Soatto, 2020;Atzmon et al., 2024) and audio (San Roman et al., 2023;Moliner et al., 2024). Similar to our method (Sec. 3), Zheng et al. (2024b) propose a frequency-masked autoencoder that extends the masked image modeling paradigm (He et al., 2022;Xie et al., 2022) to the frequency domain. AudioMAE (Huang et al., 2022) applies masked image modeling to audio spectrograms, randomly masking time-frequency bins in the audio spectrogram domain. However, our method masks latent-space frequencies.
Other works do apply the Fourier transform to hidden states, but do so as part of black-box architectural units, and focus on downstream tasks instead of directly using the latent spectra. This use of the Fourier transform has been shown to improve learning in language (Lee-Thorp et al., 2021;He et al., 2023) and vision (Rao et al., 2021;Chi et al., 2020;Guibas et al., 2021;Lin et al., 2023).
Finally, some works apply the Fourier transform post-hoc to latent states of pretrained models, choosing and interpreting latent-space frequencies. PRISM (Tamkin et al., 2020) shows that different frequency bands of language model embedding sequences are useful for different downstream tasks. In vision, Khan et al. (2017) shows that the spectra of intermediate activations in a pretrained CNN can be used to categorize scenes. These works focus on analysis, while we focus on synthesis: we can isolate frequencies in the latent representation, but also invert them and observe their realizations in the input domain. Applying frequency-domain manipulations post-hoc to pretrained representations fails to synthesize coherent audio, which we show in the DAC and RAVE baselines and our ablations (Appendix B.1). This shortcoming motivates our frequency-masking strategy during training, which deliberately encourages our latents to be manipulable in the frequency domain.
Blending. LATENTFT can blend two examples together while choosing timescales from each (by selecting latent frequencies from each example). This is like style transfer in images (Ashikhmin, 2003;Gatys et al., 2016;Johnson et al., 2016;Huang & Belongie, 2017;Deng et al., 2022;Efros & Freeman, 2023), which merges "content" from one image with the "style" from another. Applying these methods to music is challenging due to the multiscale nature of musical style, as style can refer to "high-level compositional features" or "low-level acoustic features" (Dai et al., 2018). We ameliorate this ambiguity by introducing frequency-based controls, which provide a continuous axis for specifying which timescales we want from each input. In contrast, existing works in musical style transfer focus on specific aspects of music like timbre (Huang et al., 2018;Li et al., 2024;Wang et al., 2024), musical arrangement (Cífka et al., 2020), or composition (SE, 2016). Traditional techniques are also used to blend sounds, as done in the Cross Synthesis baseline (Smith, 2011).
this section cite: ['b52', 'b93', 'b1', 'b75', 'b63', 'b32', 'b91', 'b37', 'b54', 'b33', 'b72', 'b13', 'b27', 'b57', 'b85', 'b44', 'b1', 'b26', 'b42', 'b40', 'b18', 'b20', 'b17', 'b39', 'b56', 'b88', 'b15', 'b77', 'b79']

Section: METHOD
3.1 BACKGROUND ...
x
k = 0 k = 1 k = 2 k = 3 k = 8
Figure 1: x ∈ R 16 decomposed via Eq 1.
this section cite: []

Section: Discrete
Fourier Transform. The discrete Fourier transformfoot_1 (DFT) correlates an input signal x ∈ C N with N complex sinusoidal signals, giving its spectral representation X ∈ C N . The kth DFT coefficient is given by:
X[k] = x • w k ,(DFT)
where (•) denotes the complex dot product, and w k [n] = e j(2πk/N )n denotes the kth complex sinusoid. The complex sinusoids w 1 , ..., w N form an orthogonal basis for C N , allowing the DFT to be inverted:
x = 1 N N -1 k=0 X[k]w k (IDFT)
The inverse DFT is also called the "synthesis" equation, since it expresses x as a weighted sum of complex sinusoids. To provide more concrete intuition, if x is real-valued, we can express x as the sum of real sinusoids with various frequencies k N , amplitudes A k , and phase shifts ϕ k :
x[n] = ⌊N/2⌋ k=0 A k cos 2π k N n + ϕ k (1
)
Where A k and ϕ k are both derived from the coefficient X[k], as shown in Appendix D.1. In words, the DFT can decompose a real signal into a sum of real sinusoids of different frequencies, all of which are mutually orthogonal. We show this decomposition for an example signal in Fig. 1.
this section cite: []

Section: Diffusion Autoencoders.
The diffusion autoencoder was proposed by (Preechakul et al., 2022) to harness the power of diffusion models for representation learning. During training, an encoder maps an image x 0 into a non-spatial semantic vector z sem . Then, a diffusion model (which acts as the decoder) tries to reconstruct x 0 from z sem and a noisy version of the image x τ . Diffusion autoencoders are typically trained with a MSE loss that determines how well x τ is denoised, (or equivalently, how well x 0 is reconstructed). During inference, z sem can be used to condition a generative diffusion process and produce an image.
We have three motivations for using a diffusion autoencoder. First, the decoder harnesses the generative power of a diffusion model, allowing it to generate high-quality music even when information has been removed (masked) from the latent conditioning vector. Second, since the generative process is random, one can generate multiple variations for the same input condition. Third, diffusion autoencoders have been shown to yield latent representations z sem that are semantically meaningful and linear, supporting interpolation between images and attribute manipulation. In fact, recent work shows the applicability of diffusion autoencoders to music representation learning (Pasini et al., 2024;Bindi & Esling, 2024).
User Mask Rand. Mask Encoder Input Variation Decoder Reconstruction Loss Add Noise DFT -1 DFT Latent Spec. Masked Spec.
this section cite: ['b70', 'b68', 'b4']

Section: OR OR
Figure 2: Latent Fourier Transform (LATENTFT). We encode audio (which may be represented as a waveform or spectrogram) into a series of latent vectors and compute a latent spectrum. During training (red), this spectrum is masked randomly and used to reconstruct the input. During inference (blue), the user specifies a spectral mask, which selects features from the input at specific latent frequencies and conditions a generative process.
this section cite: []

Section: METHOD OVERVIEW
Our goal is two-fold. First, we want to map an audio waveform or spectrogram x 0 into a time series of latent vectors, whose spectrum encodes semantic patterns. We refer to the DFT spectrum of this latent time series as the latent spectrum. It is important to distinguish the latent spectrum from the audible spectrum: The audible spectrum refers to the DFT spectrum of the audio waveform, and captures variations in the waveform occurring at different frequencies. In contrast, the latent spectrum captures variations in the latent time series occurring at different frequencies, which we correspond to musical patterns occurring at different timescales. Second, we should be able isolate features at selected latent frequencies and use them to generate variations, blend them with other audio clips, or hear them in isolation. These goals motivate an end-to-end encoder-decoder architecture that encodes music into latent spectra, and decodes latent spectra into music. We apply a latent Fourier transform and frequency-masking during training, shown in Alg. 1 and Fig. 2.
this section cite: []

Section: ENCODING THE LATENT SPECTRUM
Encoder. An encoder maps input music x 0 ∈ R C×T to a time series of latent vectors z ∈ R C ′ ×T ′ :
z = Enc ϕ (x 0 )(2)
Algorithm 1 Training.
Input: Audio Waveform or Spectrogram x0 1: z ← Enc ϕ (x0) 2: Z ← DFT(z) 3: η ∼ N (0, 1) ▷ Sample threshold 4: s ∼ N (0, Σ) ▷ Sample frequency bin scores 5: M ← 1s>η ▷ Get Mask 6: Z masked ← Z ⊙ M 7: z masked ← IDFT Z masked 8: Sample noise level τ ∼ p(τ ) 9: xτ ← DiffusionForward (x0, τ ) ▷ Add noise 10: x0 ← Dec θ z masked , xτ , τ ▷ Reconstruct x0 11: ℓ ← L ( x0, x0) 12: Update parameters ϕ, θ using ∇ ϕ,θ ℓ
Here, C and C ′ are the number of input and latent channels, while T and T ′ are the number of input and latent timesteps. Although T and T ′ do not have to be equal, z must have a linear temporal axis in order to produce a latent spectrum. This favors convolutional architectures or networks like the U-Net (Ronneberger et al., 2015), whose skip connections promote input-output alignment. We define f r as the latent frame rate in Hz, or the number of latent vectors (frames) needed to represent one second of audio.
Latent Fourier Transform. The latent spectrumfoot_2 refers to the DFT of the latent timeseries z, applied to each channel in the latent timeseries:
Z = DFT(z), Z ∈ C C ′ ×K
(3) Applying the DFT along the time axis of our latent sequence represents each latent channel as a sum of K = ⌊T ′ /2⌋ + 1 sinusoids (see Eq. 1). The sinusoids have K different linearly-spaced frequencies, which capture variations in each latent channel at different temporal rates. The kth sinusoid completes k cycles in T ′ latent timeframes (see Fig. 1). For instance, the 0th sinusoid is constant, the 1st sinusoid has a period of T ′ latent frames, and the 2nd sinusoid has a period of T ′ /2 latent frames. The sinusoids are also orthogonal from one another, creating an inductive bias for separating information across timescales.
Specifically, DFT(z) stores K complex coefficients indicating the amplitude and phase of each sinusoid along a length-K frequency axis. We refer to the frequency-axis of DFT(z) as the latent frequency axis, and we call points along this axis latent frequencies. Like audible frequencies, latent frequencies are described in Hz. However, 1 Hz on the latent spectrum corresponds to oscillations in the latent sequence occurring at 1 cycle per second, instead of oscillations in the audio waveform.
The kth sinusoid has a period of T ′ /k latent frames or T ′ /(kf r ) seconds, and thus a latent frequency of f k = kf r /T ′ Hz.
this section cite: ['b73']

Section: Increasing Spectral Granularity.
In practice, we zero-pad z at its end, expanding its temporal length by a factor of L. This increases the number of frequency bins by a factor of ≈ L, allowing for more spectral granularity via spectral interpolation (Smith, 2007). This is especially useful for capturing very low-frequency patterns (below 1 cycle per T ′ timeframes). We let F = ⌊LT ′ /2⌋ + 1 be the number of spectral bins (sinusoids) after zero padding z.
this section cite: ['b78']

Section: FREQUENCY MASKING
At inference, we want to select specific frequencies from the latent spectrum to generate variations from them or hear them isolation ('zoom-in' on them). This is accomplished by applying a latent spectral mask M ∈ {0, 1} F , taking Z masked = Z ⊙ M . During inference, this mask is chosen by the user. During training, this mask is randomized: First, we sample a random scalar threshold η ∼ N (0, 1), which helps decide the proportion of bins to be masked. Second, we sample s ∼ N (0 F , Σ), where s ∈ R F assigns scores to each frequency bin. Third, we set the mask to keep bins whose score is greater than the threshold, setting M = 1 s>η .
Random Threshold. Using a random threshold ensures a uniform distribution over the number of masked bins. In contrast, independently masking each frequency bin with probability p corresponds to setting a fixed threshold and Σ = I. This results in a binomial distribution over the number of masked bins, which does not reflect the inference-time distribution of user-specified masks.
this section cite: []

Section: Correlating Bins.
Instead of masking each frequency bin independently, we create a "soft grouping" between nearby frequency bins by correlating their scores. This is done by multiplying uncorrelated scores u ∼ N (0 F , I) with a radial basis function matrix K:
K i,j = c i exp - |a i -a j | p 2σ p , K ∈ R F ×F ,(4)
where a i = log(f i + ϵ) is the frequency of bin i mapped to a logarithmic axis, p, σ, and ϵ are hyperparameters, and c i normalizes each row of K to have unit ℓ 2 norm. Multiplying s = Ku results in correlated scores between frequency bins, where the amount of correlation between two frequency bins is determined by their distance on a logarithmic axis. The covariance matrix of s is Σ = KK T . Ablations (Appendix B.1) show that correlating bin scores is key to our method's performance . Intuitively, masking frequency bins independently forms speckled masks where masked bins are often adjacent to unmasked ones. The unmasked bins provide strong local cues about nearby masked ones, reducing the model's ability to fill in contiguous regions of the latent spectrum during inference. In contrast, correlated bin scores form masks with larger contiguous regions, which combats the effect of spectral leakage and better reflects inference-time, user-specified masks.
Logarithmically scaling the frequency-axis is also key to performance (Appendix B.1). This is common in audio, exemplified by the Mel scale (Stevens et al., 1937), Constant Q-Transform (Brown, 1991), and others. More generally, structured signals from images (San Roman et al., 2023) to coastlines and mountains (Bak et al., 1987) have spectra that follow a 1/f α curve. Segmenting such spectra into groups of equal width along a log-frequency axis yields groups of roughly equal energy. This motivates our logarithmic scaling, where higher frequencies are more likely to form larger groups. Lastly, normalizing the rows of K ensures equal marginal variance between every bin score s k , so that all bins have the same marginal probability of being masked for any given threshold.
this section cite: ['b83', 'b10', 'b75', 'b3']

Section: DECODING THE LATENT SPECTRUM
We transform Z masked back into the time domain by applying the inverse DFT, obtaining a frequencymasked latent sequence z masked = IDFT(Z masked ). The decoder then uses z masked to reconstruct the input x 0 from a noisy version of it (training), or to condition a diffusion process (inference).
Training. During training, we obtain a noisy version x τ of the input x 0 through a forward diffusion process. This process samples a diffusion time τ ∼ p(τ ) from a predetermined distribution and adds a τ -dependent amount of noise to x 0 . We supply z masked and x τ to the decoder, which gives an estimate of the clean input x0 :
x0 ← Dec θ z masked , x τ , τ(5)
Then, we compute a reconstruction loss ℓ = L ( x0 , x 0 ), which is used to update the parameters ϕ, θ of both the encoder and decoder. This procedure effectively trains a diffusion model, which can generate new outputs conditioned on z masked . While we do not require a particular diffusion framework, in practice we follow the ODE formulation in Karras et al. (2022). This framework preconditions the model inputs and outputs, uses approximately linear diffusion trajectories, and applies a second-order correction at each sampling step (omitted in Algs. 2 and 3 for clarity).
Algorithm 2 Conditional Generation
Input: z masked , {τi} N i=0 decreasing 1: x ∼ N (0, σ 2 max ) 2: for i ∈ {0, ..., N -1} do 3: x0 ← Dec θ z masked , x, τi 4: d ← (x -x0) /σi ▷ Deriv. of Noise Traj. 5: x ← x + (τi+1 -τi) d 6: return x Algorithm 3 Blending Input: z masked 1 , z masked 2 , {τi} N i=0 , weights α, β 1: x ∼ N (0, σ 2 max ) 2: for i ∈ {0, . . . , N -1} do 3: x(1) 0 ← Dec θ z masked 1 , x, τi 4: x(2) 0 ← Dec θ z masked 2 , x, τi 5: d1 ← (x - x(1) 0 )/σi 6: d2 ← (x - x(2) 0 )/σi 7: d ← αd1 + βd2 8: x ← x + (τi+1 -τi) d 9: return x Conditional Generation.
Our conditional generation task attempts to generate a variation of a reference song y that preserves characteristics at userspecified latent frequencies. The reference y is encoded and masked in the latent frequency domain to obtain z masked . The mask is user-specified, and typically selects low frequencies, high frequencies, or a band of intermediate frequencies. We use z masked to condition a reverse diffusion process, which iteratively denoises pure Gaussian noise to yield a new variation.
this section cite: ['b43']

Section: Blending.
Our blending task attempts to combine two musical references y 1 , y 2 into a new song that preserves characteristics from each at user-specified latent frequencies. Like before, z 1 , z 2 are obtained and masked in the latent frequency domain to get conditions z masked 1 , and z masked 2 . Here, the user specifies two masks specifying which latent frequencies to retain from each input. We obtain our blend by simulating the reverse diffusion process, at each step interpolating the derivatives induced by each condition (Alg. 3).
this section cite: []

Section: EXPERIMENTS

this section cite: []

Section: EXPERIMENTAL SETUP
Datasets and Training. We train three versions of LATENTFT with three different encoders. We use UNet and MLP encoders with a mel-spectrogram frontend, as well as a raw audio encoder that utilizes a Descript Audio Codec (DAC) (Kumar et al., 2023) frontend. These encoders are described further and compared in Appendix A.1. Each model generates mel-spectrograms, which are inverted using the BigVGAN neural vocoder (Lee et al., 2022). We train our model on MTG-Jamendo (Bogdanov et al., 2019), a large-scale collection of over 55,000 songs spanning diverse musical genres (described more in Appendix A.5) segmented into 5.9-second musical clips. Hyperparameters for the decoders and training are in Appendix A.2 and A.3, respectively.
Baselines. We compare LATENTFT to various traditional and learned methods of generating or representing audio. First, we try several generation baselines, adapting them to our task:
• Masked Token Model (Garcia et al., 2023). We use the Vampnet masked token model, which generates discrete acoustic tokens from coarse-to-fine. Vampnet is trained to predict all acoustic tokens given a random subset of them, and supports supplying arbitrary token masks during inference. For conditional generation, we select different contiguous subsets of RVQ layers to condition on, and for the blending task, we select a different layer to take from each reference.
• Guidance (Levy et al., 2023). We generate mel-spectrograms with an unconditional diffusion model. At each denoising step, we compute the DFT along the time axis of the reference spectrogram(s) and the current reconstruction x0 . We compute the loss between these DFT spectra within the selected frequency bins, using it to update the intermediate output.
• ILVR (Choi et al., 2021). We generate mel-spectrograms from an unconditional diffusion model. At each denoising step, we compute DFT spectra of the intermediate output and the reference(s) set to the current noise level. We replace selected DFT frequencies of the intermediate output with the corresponding DFT frequencies of the noisy reference(s).
• Cross Synthesis (Smith, 2011). Cross synthesis blends two sounds by replacing the spectral envelope of one sound with that of the other. We follow the implementation in Smith (2011).
In the Guidance and ILVR baselines, note that we use the spectrum of the mel-spectrogram to steer the diffusion process instead of the latent spectrum. We also attempt post-hoc frequency-domain filtering of existing representations of audio for our tasks, similar to Tamkin et al. (2020):
• DAC (Kumar et al., 2023). We encode our reference(s) using Descript Audio Codec, a popular deep neural audio codec. We frequency-mask the latent states post-quantization, and feed the filtered latent sequence to the decoder to produce audio.
• RAVE (Caillon & Esling, 2021) offers another latent representation of the audio signal, which is often manipulated in the latent space and used to generate audio (Nabi et al., 2024;Zheng et al., 2024a). Similar to DAC, we frequency-mask the latent states obtained from the RAVE encoder, then provide them to the decoder.
• Spectrogram. We filter the input mel-spectrogram representation(s) directly, by computing the DFT of the mel-spectrogram(s) along the time axis, then masking the DFT(s). We convert the filtered mel-spectrograms to audio with BigVGAN (Lee et al., 2022).
In each case, we blend by taking selected frequency components from two latent representations derived from two inputs, by adding the two frequency-masked latents together before decoding.
this section cite: ['b49', 'b53', 'b24', 'b55', 'b14', 'b79', 'b79', 'b49', 'b11', 'b65', 'b53']

Section: CONDITIONAL GENERATION
We show that LATENTFT can generate variations of a given song while preserving patterns at userspecified timescales. We take 1024 random 5.9-second clips from the MTG-Jamendo test set, ensuring each clip originates from a unique song (results on more datasets in Appendix B.2). We then generate variations of each clip, conditioning on 14 different latent frequency bands of varying widths and locations (see Appendix A.6). Good variations should adhere to the condition, preserving input characteristics at the specified timescales, and have musically coherent, high-quality audio.
Metrics. To measure adherence, we extract time-series descriptor signals (e.g., loudness curves) from both the input and generated audio. We bandpass these descriptor signals to the selected frequency band, and measure their similarity or error with standard metrics. We select four perceptually relevant time series descriptors. First, we extract loudness curves following Morrison et al. (2024), and quantify their similarity using their correlation coefficient (Kosta et al., 2016). Second, we quantify rhythmic preservation by computing onset strength envelopes (Böck & Widmer, 2013) and measuring their beat-spectral cosine similarity (Foote et al., 2002). Third, to quantify timbral preservation, we extract Mel-Frequency Cepstral Coefficients and compute Mel-Cepstral Distortion (Kominek et al., 2008). Fourth, we measure harmonic characteristics (relating to chords and music notes) by computing tonal centroid features, and quantify error using Tonnetz distance (Milne & Holland, 2016). We measure audio quality by computing the Frechet Audio Distance (Kilgour et al., 2018) between the set of generated music and the MTG-Jamendo validation set.
this section cite: ['b64', 'b48', 'b5', 'b22', 'b46', 'b62', 'b45']

Section: Results and Analysis.
We recommend listening to the qualitative results, which are available on the website 4 , and show our variations are diverse and musically interesting. Quantitative results are in Table 1. Our model outperforms all baselines in terms of adherence, indicating that the latent spectrum captures and reproduces variations in loudness, rhythm, timbre, and harmony occurring at selected timescales. We also surpass all baselines in terms of quality. Our metrics confirm that (1) previous audio generation models cannot condition on features from arbitrary timescales, and (2) previous representations of audio are not robust to post-hoc spectral modifications.
this section cite: []

Section: BLENDING
Setup. We show that LATENTFT can blend two songs together, while preserving patterns from each at user-specified latent frequencies. This application is motivated by the traditional equalizer, whose primary use is to promote coherence between tracks by emphasizing different audible frequencies from each of them. The experimental setup is similar to the conditional generation experiment. However, instead of selecting a single latent frequency band from a single song, we select two non-overlapping bands from two songs (details in Appendix A.6). We then measure the blended song's adherence to each song with respect to its selected subband, and average the two. To ensure that the blending is successful and musically coherent, we also report the FAD.
this section cite: []

Section: Analysis.
We provide examples of blending on the website, and quantitative results are shown in Table 1. The blending task requires an adherence-quality tradeoff, since adhering to both conditions perfectly may not result in pleasant audio. Since ILVR iteratively replaces frequency components of the output with those of the conditions, it has a slightly better adherence score on the timbre metric, while being worse in terms of quality. ILVR also loses to LATENTFT in user studies by a substantial margin (Fig. 3) in terms of both audio quality and ability to blend. In general, LATENTFT can better adhere to two conditions simultaneously compared to baselines, and generates higher-quality audio. The ability to adhere to disjoint latent-frequency components from two reference examples also indicates that the latent spectrum separates information by timescale to some extent.
this section cite: []

Section: LISTENING STUDY
To validate our method against human preferences, we conduct a listening study comparing LA-TENTFT and three other systems on the blending task. We choose a discrete method (the Masked Token Model baseline), a diffusion-based method (ILVR), and a traditional method (Cross Synthesis) to compare with LATENTFT. We recruited 29 musicians to complete a 12-question survey comparing every ordered pair of systems. For each question, participants first heard two randomlyselected music clips from the MTG-Jamendo test set. They then heard two blendings of the music clips, each produced by a different system. Participants rated which blending they preferred in terms of (i) audio quality and (ii) how well the clips were merged, using two separate 5-point Likert scales. Fig. 3 shows that our model outperforms the baselines on both metrics. Additional details about the listening study and statistical analyses of the results can be found in Appendix A.7.
this section cite: []

Section: HEARING IN LATENT FREQUENCIES IN ISOLATION
LATENTFT can 'zoom in' or 'boost' patterns at specific latent frequencies, analogous to how audio engineers boost various audible frequencies to identify interesting or problematic regions (Izhaki, 2017, p. 265). We show this in Fig. 4. The first spectrogram shows an electronic music clip, containing patterns at various timescales. The second spectrogram boosts latent frequencies between 0 and 1 Hz, which removes rapid drum patterns (vertical lines near the top of the spectrograms) and 1 2 3 4 5 Time (s) 512 1024 2048 4096 8192 Hz Reference 1 2 3 4 5 Time (s) 0 -1 Latent Hz 1 2 3 4 5 Time (s) 7.5 -8.5 Latent Hz Bass Bass Reduced 8 Hz Accentuated Figure 4: Isolating frequencies from an electronic music clip. We show three audio spectrograms. The second spectrogram smooths the reference spectrogram, and the third accentuates patterns occurring at 8 Hz while removing lower-frequency patterns, like the bass.
bass patterns (near the bottom), and makes the spectrogram notably smoother along the horizontal (time) axis. The third spectrogram boosts latent frequencies between 7.5 and 8.5 Hz. This accentuates a pattern in the original song occurring at 8 Hz, seen by comparing the vertical lines in the third spectrogram with those in the first. Also, the third spectrogram does not retain the rhythmic patterns of the bass, which occur below 7.5 Hz. This can be seen by comparing the lower regions of spectrograms one and three. LATENTFT allows for performing low-pass and high-pass operations on music representations while retaining musical coherence. Low-passing or high-passing spectrograms directly along their time axes cannot do this (Table 1). We achieve isolation using a self-blending procedure described in Appendix A.8. Musical concepts like genre, tempo, pitch, and chord changes are distributed across different regions of a song's latent spectrum, analogous to how different sonic characteristics occupy distinct ranges of the audible spectrum. Given a song, we generate many variants while performing a sweep through the frequencies we condition on. For each variant, we measure preservation of genre (using a classifier), chord progression, predominant pitch, and tempo, with respect to the original song. We plot how well the variation preserves these traits against the frequency we condition on, applying smoothing. Fig. 5 shows these traits are distributed across the latent spectrum differently. Genre is a more global feature; chords change at latent frequencies below 1 Hz; and predominant pitch and tempo reside at higher frequencies, tending to be multiples of the song's BPM. For this experiment, we use the GTZAN (Tzanetakis & Cook, 2002) dataset, since it contains groundtruth genre labels. More details about how these preservation curves computed are in Appendix A.9. Also, we interpret the latent spectra of more songs of various styles in Appendix B.3.
this section cite: ['b86']

Section: INTERPRETING THE LATENT SPECTRUM

this section cite: []

Section: CONCLUSION
In this work, we introduced the Latent Fourier Transform, which provides novel frequency-based controls for generative models. We showed applications in conditional generation and blending in the domain of music. Future work should include enabling real-time interactivity, or disentangling the latent spectrum along semantic axes, combining both timescale-based and semantic controls.
this section cite: []

Section: References
Ref_id:b0 Title: Generating music from text Year: (2023)
Ref_id:b1 Title: Edify image: High-quality image generation with pixel space laplacian diffusion models Year: (2003)
Ref_id:b2 Title: An architecture for deep, hierarchical generative models Year: (2016)
Ref_id:b3 Title: Self-organized criticality: An explanation of the 1/f noise Year: (1987)
Ref_id:b4 Title: Unsupervised composable representations for audio Year: (2024)
Ref_id:b5 Title: Maximum filter vibrato suppression for onset detection Year: (2013-09)
Ref_id:b6 Title: Essentia: an open-source library for sound and music analysis Year: (2013)
Ref_id:b7 Title: The mtg-jamendo dataset for automatic music tagging Year: (2019)
Ref_id:b8 Title: Audiolm: a language modeling approach to audio generation Year: (2023)
Ref_id:b9 Title: Efficient parallel audio generation Year: (2023)
Ref_id:b10 Title: Calculation of a constant q spectral transform Year: (1991)
Ref_id:b11 Title: Rave: A variational autoencoder for fast and high-quality neural audio synthesis Year: (2021)
Ref_id:b12 Title: Enhancing novelty in text-to-music generation using beat-synchronous mixup strategies Year: (2024)
Ref_id:b13 Title: Fast fourier convolution Year: (2020)
Ref_id:b14 Title: Conditioning method for denoising diffusion probabilistic models Year: (2021)
Ref_id:b15 Title: Groove2groove: One-shot music style transfer with supervision from synthetic data Year: (2020)
Ref_id:b16 Title: Simple and controllable music generation Year: (2023)
Ref_id:b17 Title: Music style transfer: A position paper Year: (2018)
Ref_id:b18 Title: Stytr2: Image style transfer with transformers Year: (2022)
Ref_id:b19 Title: Circnn: accelerating and compressing deep neural networks using block-circulant weight matrices Year: (2017)
Ref_id:b20 Title: Image quilting for texture synthesis and transfer Year: (2023)
Ref_id:b21 Title: Stable audio open Year: (2025)
Ref_id:b22 Title: Audio retrieval by rhythmic similarity Year: (2002)
Ref_id:b23 Title: Riffusion -Stable diffusion for real-time music generation Year: (2022)
Ref_id:b24 Title: Vampnet: Music generation via masked acoustic token modeling Year: (2023)
Ref_id:b25 Title: Sketch2sound: Controllable audio generation via time-varying signals and sonic imitations Year: (2025)
Ref_id:b26 Title: Image style transfer using convolutional neural networks Year: (2016)
Ref_id:b27 Title: Anima Anandkumar, and Bryan Catanzaro. Adaptive fourier neural operators: Efficient token mixers for transformers Year: (2021)
Ref_id:b28 Title: A latent variable model for natural images Year: (2016)
Ref_id:b29 Title: Music tagging with classifier group chains Year: (2025)
Ref_id:b30 Title: Enabling factorized piano music modeling and generation with the maestro dataset Year: (2018)
Ref_id:b31 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b32 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b33 Title: Fourier transformer: Fast long range modeling by removing sequence redundancy with fft operator Year: (2023)
Ref_id:b34 Title: Gaussian error linear units (gelus) Year: (2016)
Ref_id:b35 Title: Cnn architectures for large-scale audio classification Year: (2017)
Ref_id:b36 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b37 Title: Masked autoencoders that listen Year: (2022)
Ref_id:b38 Title: Noise2music: Text-conditioned music generation with diffusion models Year: (2023)
Ref_id:b39 Title: Timbretron: A wavenet (cyclegan (cqt (audio))) pipeline for musical timbre transfer Year: (2018)
Ref_id:b40 Title: Arbitrary style transfer in real-time with adaptive instance normalization Year: (2017)
Ref_id:b41 Title: Mixing audio: concepts, practices, and tools Year: (2017)
Ref_id:b42 Title: Perceptual losses for real-time style transfer and super-resolution Year: (2016)
Ref_id:b43 Title: Elucidating the design space of diffusionbased generative models Year: (2022)
Ref_id:b44 Title: Scene categorization with spectral features Year: (2017)
Ref_id:b45 Title: Fr\'echet audio distance: A metric for evaluating music enhancement algorithms Year: (2018)
Ref_id:b46 Title: Synthesizer voice quality of new languages calibrated with mean mel cepstral distortion Year: (2008)
Ref_id:b47 Title: Diffwave: A versatile diffusion model for audio synthesis Year: (2020)
Ref_id:b48 Title: Mapping between dynamic markings and performed loudness: a machine learning approach Year: (2016)
Ref_id:b49 Title: Highfidelity audio compression with improved rvqgan Year: (2023)
Ref_id:b50 Title: The measurement of observer agreement for categorical data Year: (1977)
Ref_id:b51 Title: Highfidelity music vocoder using neural audio codecs Year: (2025)
Ref_id:b52 Title: Single-image depth estimation based on fourier domain analysis Year: (2018)
Ref_id:b53 Title: A universal neural vocoder with large-scale training Year: (2022)
Ref_id:b54 Title: Mixing tokens with fourier transforms Year: (2021)
Ref_id:b55 Title: Controllable music production with diffusion models and guidance gradients Year: (2023)
Ref_id:b56 Title: Music style transfer with time-varying inversion of diffusion models Year: (2024)
Ref_id:b57 Title: Deep frequency filtering for domain generalization Year: (2023)
Ref_id:b58 Title: Audioldm: Text-to-audio generation with latent diffusion models Year: (2023)
Ref_id:b59 Title: Fast training of convolutional networks through ffts Year: (2013)
Ref_id:b60 Title: Eric Battenberg, and Oriol Nieto. librosa: Audio and music signal analysis in python Year: (2015)
Ref_id:b61 Title: Sdedit: Guided image synthesis and editing with stochastic differential equations Year: (2021)
Ref_id:b62 Title: Empirically testing tonnetz, voice-leading, and spectral models of perceived triadic distance Year: (2016)
Ref_id:b63 Title: A diffusion-based generative equalizer for music restoration Year: (2024)
Ref_id:b64 Title: Fine-grained and interpretable neural speech editing Year: (2024)
Ref_id:b65 Title: Embodied exploration of deep latent spaces in interactive dance-music performance Year: (2024)
Ref_id:b66 Title: Ditto: Diffusion inference-time t-optimization for music generation Year: (2017)
Ref_id:b67 Title: A music generation model that listens Year: (2024)
Ref_id:b68 Title: Music2latent: Consistency autoencoders for latent audio compression Year: (2024)
Ref_id:b69 Title: Analyzing the effect of k-space features in mri classification models Year: (2024)
Ref_id:b70 Title: Diffusion autoencoders: Toward a meaningful and decodable representation Year: (2022)
Ref_id:b71 Title: Marios Savvides, and Bhiksha Raj. Efficient autoregressive audio modeling via next-scale prediction Year: (2024)
Ref_id:b72 Title: Global filter networks for image classification Year: (2021)
Ref_id:b73 Title: U-net: Convolutional networks for biomedical image segmentation Year: (2015)
Ref_id:b74 Title: Melody extraction from polyphonic music signals: Approaches, applications, and challenges Year: (2014)
Ref_id:b75 Title: From discrete tokens to high-fidelity audio using multi-band diffusion Year: (2023)
Ref_id:b76 Title: Moûsai: Efficient text-tomusic diffusion models Year: (2024)
Ref_id:b77 Title: Musical style modification as an optimization problem Year: (2016)
Ref_id:b78 Title: Mathematics of the discrete Fourier transform (DFT): with audio applications Year: (2007)
Ref_id:b79 Title: Spectral audio signal processing Year: (2011)
Ref_id:b80 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b81 Title: Ladder variational autoencoders Year: (2016)
Ref_id:b82 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b83 Title: A scale for the measurement of the psychological magnitude pitch Year: (1937)
Ref_id:b84 Title: Wave-u-net: A multi-scale neural network for end-to-end audio source separation Year: (2018)
Ref_id:b85 Title: Language through a prism: A spectral approach for multiscale language representations Year: (2020)
Ref_id:b86 Title: Musical genre classification of audio signals Year: (2002)
Ref_id:b87 Title: Neelesh Ramachandran, and Mert Pilanci. Subtractive training for music stem insertion using latent diffusion models Year: (2025)
Ref_id:b88 Title: A training-free approach for music style transfer with latent diffusion models Year: (2024)
Ref_id:b89 Title: Music controlnet: Multiple time-varying controls for music generation Year: (2024)
Ref_id:b90 Title: Group normalization Year: (2018)
Ref_id:b91 Title: Simmim: A simple framework for masked image modeling Year: (2022)
Ref_id:b92 Title: Discrete diffusion model for text-to-sound generation Year: (2023)
Ref_id:b93 Title: Fda: Fourier domain adaptation for semantic segmentation Year: (2020)
Ref_id:b94 Title: Soundstream: An end-to-end neural audio codec Year: (2021)
Ref_id:b95 Title: Learning hierarchical features from generative models Year: (2017)
Ref_id:b96 Title: A mapping strategy for interacting with latent audio synthesis using artistic materials Year: (2024)
Ref_id:b97 Title: Mfae: Masked frequency autoencoders for domain generalization face anti-spoofing Year: (2024)
