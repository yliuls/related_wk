Title: Efficient and Separate Authentication Image Steganography Network
Abstract: Image steganography hides multiple images for multiple recipients into a single cover image. All secret images are usually revealed without authentication, which reduces security among multiple recipients. It is elegant to design an authentication mechanism for isolated reception. We explore such mechanism through sufficient experiments, and uncover that additional authentication information will affect the distribution of hidden information and occupy more hiding space of the cover image. This severely decreases effectiveness and efficiency in large-capacity hiding. To overcome such a challenge, we first prove the authentication feasibility within image steganography. Then, this paper proposes an image steganography network collaborating with separate authentication and efficient scheme. Specifically, multiple pairs of lock-key are generated during hiding and revealing. Unlike traditional methods, our method has two stages to make appropriate distribution adaptation between locks and secret images, simultaneously extracting more reasonable primary information from secret images, which can release hiding space of the cover image to some extent. Furthermore, due to separate authentication, fused information can be hidden in parallel with a single network rather than traditional serial hiding with multiple networks, which can largely decrease the model size. Extensive experiments demonstrate that the proposed method achieves more secure, effective, and efficient image steganography. Code is available at https://github.com/Revive624/Authentication- Image-Steganography.

Section: Introduction

this section cite: []

Section: Overview and problems
Image steganography involves embedding secret images within a cover image. The recipients can reveal hidden images from the stego image. For effective image steganography, the stego image should closely resemble the cover image, and recovered images must closely match the original secret images. Due to confidentiality, image steganography has been widely employed in digital watermarking (Zhang et al., 2024;Wang et al., 2023), Internet of Things (IoT) (Khari et al., 2020), military communications (Pratik et al., 2022), quantum computing (Bharatwaj & Hasabnis, 2024), healthcare (Issac & Kumar, 2023), and various domains.
Learning-based image steganography has become mainstream. Deep Neural Networks (DNNs) (Baluja, 2017;2019;Zhang et al., 2020a) are introduced in image steganography, hiding secret images in a cover image. Some GANsbased methods (Hayes & Danezis, 2017;Chen et al., 2022;Kishore et al., 2022;Li et al., 2023) are proposed with a generator for hiding and a discriminator for identifying. Some methods (Jing et al., 2021;Lu et al., 2021;Guan et al., 2022) regard concealing and revealing as a pair of reversible processes, and introduce Invertible Neural Networks (INNs) to hide and reveal images with shared parameters. Recently, diffusion-based models (Yu et al., 2024;Yang et al., 2024) have provided secure and robust steganography. However, current methods still suffer from three major defects.
(1) Lack of authentication. As shown in Figure 1, all secret images are revealed at once, without verifying the identity of recipients. This defect among multiple recipients can lead to unauthorized access and serious information leakage, which fails to meet practical application requirements.
(2) Low quality when hiding multiple images. Largecapacity hiding methods (Lu et al., 2021;Guan et al., 2022) hide all secret images in the limited space of a cover image. When the space is insufficient to accommodate more secret information, the network may either sacrifice the space of the cover image or lose secret information, both reducing transmission effectiveness.
(3) Large increase in model size and computational cost. For traditional methods of serial hiding (Guan et al., 2022;Zhou et al., 2024), multiple networks are required to be assembled to hide secret images. This results in a linear growth in model size and computational cost as the number of secret images increases.
this section cite: ['b38', 'b27', 'b16', 'b24', 'b4', 'b14', 'b2', 'b3', 'b12', 'b6', 'b18', 'b21', 'b15', 'b22', 'b11', 'b35', 'b34', 'b22', 'b11', 'b11', 'b39']

Section: Exploration and Challenges
In order to address non-authentication and achieve isolated reception, it is elegant to introduce an authentication mechanism. Following IIS (Zhou et al., 2024), we establish an authentication-based image steganography network. Through embedding additional lock information in the cover image, a corresponding key is required to recover the secret images. Furthermore, an authentication-free network is built, which removes authentication while preserving the hiding network. Based on the above two methods, we conduct comparison experiments to evaluate the effectiveness of authentication and its influence on hiding process on DIV2K dataset.
In such comparison, through a large number of samples and lots of statistics, Jensen-Shannon (JS) divergence and Peak Signal-to-Noise Ratio (PSNR) are calculated to quantify the global and local differences between cover and stego images generated by authentication-based method (blue) and authentication-free variant (red), as shown in Figure 2. From (a), blue spots show a general distribution of higher overall JS divergence and lower PSNR. This indicates that authentication-based method may cause more significant alterations in the global distribution and also reduce the local similarity to the cover images. From (b), authenticationbased method shows a great decline in PSNR and SSIM on average, indicating that the additional lock information will decrease the similarity between cover and stego images.
Based on the insights gained from the results, our exploration of authentication mechanism highlights two crit- ical challenges for building authentication-based image steganography, which further motivates our improvement on embedding lock information.
(1) Embedding authentication information while maintaining the quality of stego and revealed secret images.
Due to the inconsistent distribution of secret images and locks, the embedded information usually occupies considerable hiding space of the cover image, resulting in a great drop in the quality of both stego and revealed secret images and reducing the transmission effectiveness.
(2) Integrating authentication while limiting the model size. Compared to authentication-free method, the parameter number of authentication-based method grows from 16.3M to 19.9M, shown in Figure 2 (b). The extra parameters bring additional computational cost and severely reduce transmission efficiency.
this section cite: ['b39']

Section: Solution

this section cite: []

Section: To overcome the challenges of existing methods, we propose an Efficient and Separate Authentication Image Steganography Network (AIS) with two collaborative stages, consisting of an Invertible Authentication Network (IAN) and an Invertible Hiding Network (IHN), respectively.
In the first stage, IAN introduces an authentication mechanism. The mechanism embeds locks in secret images to ensure that only recipients with correct keys can access the hidden secret images. Locks and keys are generated from a dynamic lock-key generation strategy by learning features of both the cover and secret images. What is more, different from traditional one-stage methods, our IAN fuses the locks and secret images. Then, such fused information will be embedded in the cover image in the second stage (IHN). This will produce a better distribution adaptation and may free up space for hiding information. Additionally, IAN extracts more reasonable primary information from secret images. This reduces the secret information to be hidden.
All above strategies can both achieve isolated reception and produce high-quality stego and revealed secret images.
In the second stage, IHN utilizes the fused lock and secret information with better distribution adaptation in the hiding process to produce high-quality stego images. Furthermore, due to separate authentication, IHN can hide all secret images in parallel. This approach only requires training a single hiding network, unlike serial hiding methods that require an extra network for each secret image. The main contributions of our method are summarized as follows:
• We propose an Authentication Invertible Steganography (AIS) collaboratively consisting of an Invertible Authentication Network (IAN) and an Invertible Hiding Network (IHN). The network enables isolated reception and enhances security among multi-recipients.
• The proposed IAN can generate pairs of lock-key and fuse the locks and secret images to make distribution adaptation. This will decrease the locks occupation of the hiding space in cover images.
• The proposed IAN also contains learnable mappings to extract more reasonable primary information from secret images, producing high-quality stego and revealed secret images on the basis of isolated reception.
• The proposed IHN employs the fused secret and lock information to produce stego images. IHN also allows for efficient parallel hiding of secret images, requiring only a single network to be trained, significantly reducing the model size and computational cost.
this section cite: []

Section: Related Work

this section cite: []

Section: Authentication-Free Image Steganography
Image steganography aims to conceal secret images within a single cover image, allowing recipients to accurately reveal the hidden information. Baluja (Baluja, 2017;2019) proposed the first deep neural network for image steganography. HiNet (Jing et al., 2021) used INNs for reversible hiding and revealing in frequency domain. ISN (Lu et al., 2021) extended INNs to parallel processing in spatial domain for multiple secret images. DeepMIH (Guan et al., 2022) explored a serial hiding strategy with an importance map for optimized spatial utilization. InvMIHNet (Chen et al., 2024) achieved large-capacity hiding by splicing multiple images into a single secret image. Recently, CRoSS (Yu et al., 2024) explored DDIM Inversion to transform secret images into stego images without a cover image. However, these methods do not support recipient verification and reveal all secret images simultaneously. This increases the risk of unauthorized access and reduces practical flexibility.
this section cite: ['b2', 'b3', 'b15', 'b22', 'b11', 'b7', 'b35']

Section: Authentication-Based Image Steganography
Image steganography with certification is challenging, as it requires embedding authentication information while minimizing the impact on stego and revealed secret images. Kweon et al. (Kweon et al., 2021) proposed a key mechanism based on an encoder-decoder structure. IIS (Zhou et al., 2024) dynamically generated global and local keys for each image, embedding fused keys into stego images via an INN structure for authentication. DiffStega (Yang et al., 2024) achieved coverless steganography through diffusion models, where an image prompt serves as a private key guiding the revealing. Although these methods enable authentication, the authentication information often degrades the quality of hiding and recovery, reducing transmission effectiveness. Furthermore, these methods typically have large model sizes, struggling to balance verifiability and efficiency in large-capacity hiding.
this section cite: ['b20', 'b39', 'b34']

Section: Invertible Neural Networks
Invertible Neural Networks (INNs) establish a bijective mapping between data distribution p x and latent distribution p z . Dinh et al. (Dinh et al., 2015;2017) introduced coupling layers in generative flow models. Gilbert et al. (Gilbert et al., 2017) explored the reversibility of INNs. Kingma et al. (Kingma & Dhariwal, 2018) improved generative flow models using invertible 1×1 convolutional layers. Xiao et al. (Xiao & Liu, 2020) proposed matrix exponential coupling layers for improved density estimation performance. Ardizzone et al. (Ardizzone et al., 2019) proposed Conditional Invertible Neural Networks (CINNs), where a condition guides the generation. Koehler et al. (Koehler et al., 2021) theoretically explored the depth and condition of normalizing flows, which highlights the trade-off between authentication and generation quality in image steganography.
To address these challenges, we propose distribution adaptation and secret information extraction. This enables isolated reception while ensuring high quality for both stego and revealed secret images. Separate authentication promotes parallel hiding, thereby reducing the model size.
this section cite: ['b8', 'b2', 'b10', 'b17', 'b31', 'b0', 'b19']

Section: Method
In this part, we first demonstrate the feasibility of incorporating authentication in our two-stage image steganography.
Then, we detail the structure of our proposed Efficient and Separate Authentication Image Steganography Network (AIS), consisting of an Invertible Authentication Network (IAN) and an Invertible Hiding Network (IHN), as shown in Figure 3. IAN embeds locks in secret images and verifies keys during the revealing process, while extracting more reasonable primary information of secret images. IHN hides extracted information in the cover image and reveals it from Harr Downsampling AInvBlock 𝜶 𝜷 𝜸 𝒆 AInvBlock C C … 𝑬𝒙𝒕𝒓𝒂𝒄𝒕 𝟏,𝟐,𝟑 𝑺𝒆𝒄𝒓𝒆𝒕 𝟏,𝟐,𝟑 𝑹 𝒉 𝟏,𝟐,𝟑 𝑪𝒐𝒗𝒆𝒓 DWT HInvBlock 𝜶 𝜷 𝜸 𝒆 HInvBlock … IWT 𝑺𝒕𝒆𝒈𝒐 𝑹 𝒐 Harr Upsampling AInvBlock 𝜶 𝜷 𝜸 𝒆 AInvBlock C C … 𝑹𝒆𝒗𝑬𝒙𝒕𝒓𝒂𝒄𝒕 𝟏,𝟐,𝟑 𝑹𝒆𝒗𝒆𝒂𝒍 𝟏,𝟐,𝟑 the stego image. Locks and keys for IAN are generated by a dynamic generation module, a simplified UNet structure (Ronneberger et al., 2015). Further details are provided in Appendix B.
this section cite: ['b25']

Section: Feasibility of Authentication
Image steganography involves hiding and revealing processes, which can be modeled as a normalizing flow f (Dinh et al., 2015). x denotes the data domain of secret images and z denotes the latent domain. The probability distribution p x (x) of the secret images can be derived via the change-of-variable formula:
px (x; θ) = p z (f θ (x)) • det ∂f θ (x) ∂x ,(1)
where det(•) denotes the determinant. Given data samples {x i } N i=1 , the parameters θ can be optimized by maximizing log likelihood loss:
L θ = N i=1 log(p z (f θ (x i )))+log det ∂f θ (x i ) ∂x i .(2)
To incorporate authentication information, we adopt the conditional normalizing flow framework (Ardizzone et al., 2019;2021;Wen et al., 2023). The lock and key are denoted by c and c ′ , respectively. The forward mapping is defined as: z = f θ (x, c) and the backward mapping is defined as: z = f θ (x, c ′ ). The change-of-variable formula is:
px|c (x|c; θ) = p z (f θ (x, c)) • det ∂f θ (x, c) ∂x .(3)
Through the flow, the distribution of the secret images is influenced by the authentication information of the lock.
Through training, x can be shifted away from the distribution of x when given the wrong key, while x can approximate the distribution of x when given the correct key. A detailed proof is provided in Appendix A.2.
this section cite: ['b8', 'b0', 'b28']

Section: Overall framework
In the forward hiding process, IAN first generates locks with the dynamic lock-key generation module. Then, locks and secret images are fused to make distribution adaptation. Meanwhile, IAN extracts more reasonable primary information from the secret images. The cover image is transformed into the frequency domain to match the extracted information in size. Both the cover image and extracted information are fed into the IHN to produce the stego image, while redundant information is omitted during transmission. In the backward revealing process, following IIS (Zhou et al., 2024), redundancy is restored using a Redundancy Prediction Module to better fit the latent distribution. The primary information is recovered through the reverse process of IHN. Then, a key is generated by the dynamic lock-key generation module, and used in the IAN for verification, gradually decoupling the complete secret image.
this section cite: ['b39']

Section: Invertible Authentication Network
IAN comprises several authentication invertible blocks, a Haar Downsampling Module and a Haar Upsampling Module. To enable isolated reception, each secret image is processed independently rather than as a combined input.
In the forward hiding process, the Haar Downsampling Module transforms a RGB secret image S o ∈R B×3×H×W from the spatial domain to the frequency domain S f ∈R B×12×(H/2)×(W/2) . S f is then decomposed into a low-frequency component S l ∈R B×3×(H/2)×(W/2) and a high-frequency component S h ∈R B×9×(H/2)×(W/2) , which are fed into the authentication invertible blocks. The transformation in the i th block is described by:
x i+1 = x i + α(y i ), y i+1 = y i ⊗ exp(β(x i+1 , L)) + γ(x i+1 , L),(4)
where x and y represent the low-frequency and highfrequency components, L denotes the lock, ⊗ is Hadamard product, exp(•) is exponential function. The functions α(•), β(•) and γ(•) are learnable mappings based on DenseNet structures (Huang et al., 2017). Unlike general INNs, IAN concatenates x i+1 and the lock L along the channel dimension as inputs to these mappings. Through these learnable mappings, IAN makes distribution adaptation between locks and secret images, outputting the extracted information S c with 3 channels and redundant information R h .
In the backward revealing process, since R h in the forward process is omitted, an auxiliary variable is introduced. Following (Xiao et al., 2023), a latent variable R n ∈R B×9×(H/2)×(W/2) is sampled from a Gaussian distribution. The invertible blocks iteratively recover the secret images from this latent variable under the key K for verification. The reverse transformation is given by:
y i = (y i+1 -γ(x i+1 , K)) ⊗ exp(-β(x i+1 , K)), x i = x i+1 -α(y i ),(5)
where x represents the revealed extracted information, y denotes the auxiliary latent variable. The parameters of α(•), β(•) and γ(•) are shared between the forward and backward processes. The key K must closely match the embedded lock L for accurate decoupling of the secret image. Otherwise, deviations in the key will introduce errors, causing the revealed secret image to diverge from the original secret image. Given the correct key, the revealed secret image S ro is reconstructed using the Haar Upsampling Module.
this section cite: ['b13', 'b32']

Section: Invertible Hiding Network
The Invertible Hiding Network (IHN) consists of several hiding invertible blocks, a Discrete Wavelet Transformation (DWT) module and an Inverse Discrete Wavelet Transformation (IWT).
In the forward hiding process, the RGB cover image C o ∈R B×3×H×W is transformed from the spatial domain to the frequency domain C f ∈R B×12×(H/2)×(W/2) via DWT. This transformation not only aligns C f with S c in dimensions, but also enhances hiding quality and robustness against steganalysis (Guan et al., 2022). Thanks to separate authentication in IAN, IHN can hide secret images in parallel. The secret image components S 1,2,3 c are concatenated along the channel dimension and, together with C f , are fed into the invertible blocks. The computation within the i th invertible block is described as:
x i+1 = x i + α(y i ), y i+1 = y i ⊗ exp(β(x i+1 )) + γ(x i+1 ),(6)
where x and y represent the cover and secret information, respectively. The functions α(•), β(•) and γ(•) share the same structure as those in the IAN. The forward process produces a stego image T f and redundant information R o , and T f is restored to the spatial domain T via IWT.
In the backward revealing process, following (Zhou et al., 2024), a Redundancy Prediction Module (RPM) is employed to ensure reversibility and enhance the quality of revealed secret images. RPM is a learnable module with a residual structure as described in (Mou et al., 2023). It adapts an auxiliary variable R p from the stego image T to closely approximate R o . This ensures the reversibility of the invertible blocks and preserves high fidelity in the revealed secret images. Using R p and the frequency-domain stego image T f , the reverse computation is defined as:
y i = (y i+1 -γ(x i+1 )) ⊗ exp(-β(x i+1
)),
x i = x i+1 -α(y i ),(7)
where the invertible blocks gradually decouple the extracted information from the stego image. Channel split outputs the revealed extracted information S rc .
this section cite: ['b11', 'b39', 'b23']

Section: Loss Functions
Stego images should closely resemble the cover images.
The hiding loss is defined as:
L h = L 2 (C o , T ) + L 2 (C l , T l ),(8)
where L 2 represents the Mean Square Error (MSE) loss, T l and C l are the low-frequency components of T and C o after DWT. The low-frequency loss part enhances visual similarity and improves resistance to steganalysis (Guan et al., 2022).
Revealed secret images should closely match the original secret images for effective transmission. Thus, the revealing loss consists of:
L rc = N i=1 L 2 (S i c , S i rc ), L r = N i=1 L 2 (S i o , S i ro ) + N i=1 J S(S i o , S i ro ),(9)
where J S denotes Jensen-Shannon Divergence loss, which further improves the quality of S ro (Chen et al., 2024). Due to strict authentication, keys should closely match the generated locks. This is represented by the key loss:
L k = N i=1 L 1 (L i , K i ), (10
)
where L 1 represents the Mean Absolute Error (MAE) loss.
The authentication mechanism requires that entering a wrong key should cause the revealed information to diverge from the secret image. This is enforced by the triplet loss:
L t = N i=1 max{0, L 2 (S i o , S i ro )-L 2 (S i o , S i rn )+margin},(
11) where S rn is a negative sample generated using a random key during the revealing process, and margin is a constant set to 1 (Schroff et al., 2015).
Following literature (Zhou et al., 2024), a redundancy loss is defined to ensure similarity between the predicted redundancy and the forward output of the redundant information:
L p = N i=1 L 2 (R i o , R i p ).(12)
The overall training loss is the weighted sum of the above losses:
L = λ 1 L h + λ 2 L rc + λ 3 L r + L k + L t + L p ,(13)
where λ 1 , λ 2 , λ 3 are hyper-parameters that balance the contribution of each loss term.
this section cite: ['b11', 'b7', 'b26', 'b39']

Section: Experiments
We compare our method with three baselines: ISN (Lu et al., 2021), DeepMIH (Guan et al., 2022) and IIS (Zhou et al., 2024) in terms of PSNR, SSIM and LPIPS, on DIV2K dataset and ImageNet dataset. The training settings and details are provided in Appendix C.
this section cite: ['b22', 'b11']

Section: Quality Analysis
Quantitative Results. Table 1 presents quantitative comparisons. The results for the revealed secret images are reported as averages. For clarity, LPIPS values are scaled by 10 3 . On DIV2K dataset, our AIS achieves superior performance across all metrics. For stego images, PSNR, SSIM, and LPIPS show improvements of 5.062 dB, 0.096, and 0.512, respectively. For the revealed secret images, PSNR, SSIM, and LPIPS are optimized by 2.063 dB, 0.046, and 0.047, respectively. On ImageNet dataset, our AIS achieves SSIM improvements of 0.043 and 0.026 for stego images and revealed secret images, respectively. Similar enhancements are observed in PSNR and LPIPS.  Qualitative Results. Figure 4 presents the visual comparisons between our method and other baselines. Residual errors, amplified by a factor of 10, are used to emphasize differences between the generated images and the ground-truth images. From the results, significant color distortions appear in the stego images and revealed secret images from other methods. This indicates that a single cover image cannot effectively accommodate information from multiple secret images, leading to color loss. In contrast, our AIS method, which hides only extracted information, narrows the gap between the stego and cover images while preserving the fine details of the secret images.
The above results demonstrate that our AIS method significantly improves both the stego and revealed secret images. Our method effectively extracts more reasonable primary features of secret images, reducing the information needed for hiding and revealing. Meanwhile, our AIS integrates the lock with the secret images, partially avoiding embedding information that has an inconsistent distribution in the cover image. This may reduce the occupation of hidden space. Both allow a single cover image to accommodate more secret images, while preserving sufficient information for high-quality restoration.
this section cite: []

Section: Security Analysis

this section cite: []

Section: EFFECTIVENESS OF AUTHENTICATION
To demonstrate the effectiveness of our proposed authentication mechanism, we simulate attempts to reveal secret images using random keys on DIV2K dataset. As shown in Figure 5, we use incorrect keys, such as all zeros, all ones, Gaussian distribution, uniform distribution, and a key of another secret image. The restored images are almost indistinguishable. Clearly, information revealed using randomly generated keys is almost meaningless among unauthorized recipients.
These results imply that the authentication mechanism effectively identifies recipients. Secret images are restored independently with correct keys. When using a forged key, the revealed secret images become severely corrupted. This enhances the system's security against attacks and provides flexibility in sending different images to different recipients.
this section cite: []

Section: RESISTANCE AGAINST STEGANALYSIS
Steganalysis methods aim to detect hidden information in stego images. A robust steganography method must perform well in both quality and resistance to steganalysis.
Otherwise, attackers can easily identify stego images, compromising transmission confidentiality and integrity.
this section cite: []

Section: Detection of Stego Image.
We test our method against three state-of-the-art steganalysis methods, SRNet (Boroumand et al., 2019), ZhuNet (Zhang et al., 2020b), and LWENet (Weng et al., 2022). 1000 cover-stego image pairs are generated using baselines and our AIS on ImageNet dataset. Table 2 shows that our AIS achieves the lowest detection accuracy among all methods, decreasing by 8.15%, 6.40% and 10.45%. This improvement is due to the proposed IAN in our method, which reduces the amount of hidden information, making stego and cover images harder to distinguish.
this section cite: ['b5', 'b29']

Section: Detection of Secret Information.
To evaluate potential information leakage, we employ ManTraNet (Wu et al., 2019), a method designed to detect the locations of hidden data within an image. The white regions in the mask highlight detected anomalies. Figure 6 shows fewer abnormal areas detected by ManTraNet in stego images of our method. While the white areas detected in other methods reveal contours of hidden secret images, those detected in our AIS hardly reveal any meaningful information. These results demonstrate superior security of our method in preventing information leakage. In terms of FLOPs, our AIS shows superior computational efficiency, saving an average of 252.35G compared to ISN. Additionally, our method significantly reduces inference time compared to DeepMIH and IIS. Due to the superior design of AIS, parallel hiding significantly reduces model size and computational cost. As the number of secret images increases, our method retains its efficiency advantage. This scalability demonstrates its suitability for large-capacity steganography applications.
this section cite: ['b30']

Section: Ablation Studies

this section cite: []

Section: EXTRACTION CHANNELS
The proposed IAN extracts features of 3 channels from secret images, focusing on retaining the primary information.
To assess the performance among different channel numbers, we conduct ablation experiments on DIV2K dataset. The results summarized in Table 3 show that 3 channels yield the best quality for both stego and revealed secret images. This indicates that increasing the channel number intensifies recovery errors caused by the proposed IHN, leading to significant information loss and degraded image quality in  scenarios with more channels.
this section cite: []

Section: EFFECTIVENESS OF DISTRIBUTION ADAPTATION
To demonstrate the IAN's effectiveness of distribution adaptation in our AIS method (blue), we compare it with an authentication-free variant (red) modified from AIS and a one-stage authentication method on DIV2K dataset. As illustrated in Figure 7 (a), our method achieves a lower overall JS Divergence and higher PSNR and SSIM. This suggests that distribution adaptation partially avoids embedding information with inconsistent distribution in the cover image. Moreover, due to parallel hiding, only a small number of parameters has been increased. (b) shows the mean and standard deviation to compare the information distribution between the one-stage method and our two-stage AIS method. In one-stage method, the distribution of locks (orange) and secret images (green) is inconsistent, whereas our AIS (purple) achieves a more consistent distribution. This indicates that the proposed IAN adapts secret images and locks to fused information with a consistent latent distribution. Through such adaptation, AIS may avoid the additional lock information to occupy the hiding space, enhancing the quality of both stego and revealed secret images.
this section cite: []

Section: EFFECTIVENESS OF PRIMARY INFORMATION
To evaluate the effectiveness of the primary information extracted by the proposed IAN, we compare our method with one-stage method without extraction on DIV2K dataset. (a) shows the difference of the hiding and revealing processes between one-stage method and our two-stage AIS. This instance displays that the secret image can be restored from the primary information. In (b), a boxplot compares information entropy between extracted information (blue) and secret images (red), and another boxplot compares PSNR of revealed secret images between one-stage method (red) and our AIS (blue). Compared to one-stage method, our AIS decreases the average information entropy by 1.85, but improves the average PSNR by 2.086 db. The results indicates that our method may filter the redundant information to some extent, and contains less redundant information, extracting more reasonable primary information while pre-serving the quality of the revealed secret images.
this section cite: []

Section: Conclusion
This paper demonstrates the feasibility of an authentication mechanism and proposes an Efficient and Separate Authentication Image Steganography (AIS) method. AIS embeds locks in secret images. Distribution adaptation partially releases the space in cover images and extracts primary information for hiding, enhancing the quality of both stego and secret images. A correct key is needed to recover the secret image, achieving isolated reception among different recipients. Due to separate authentication, only a single network needs to be trained, significantly limiting the model size. The specially designed two-stage method enables secure, effective, efficient, and flexible image steganography.
this section cite: []

Section: References
Ref_id:b0 Title: Guided image generation with conditional invertible neural networks Year: (2019)
Ref_id:b1 Title: Conditional invertible neural networks for diverse image-to-image translation Year: (2021)
Ref_id:b2 Title: Hiding images in plain sight: Deep steganography Year: (2017)
Ref_id:b3 Title: Hiding images within images Year: (2019-02)
Ref_id:b4 Title: Steganography in the quantum era Year: (2024)
Ref_id:b5 Title: Deep residual network for steganalysis of digital images Year: (2019-05)
Ref_id:b6 Title: Hiding images in deep probabilistic models Year: (2022)
Ref_id:b7 Title: Invertible mosaic image hiding network for very large capacity image steganography Year: (2024)
Ref_id:b8 Title: Nice: Non-linear independent components estimation Year: (2015)
Ref_id:b9 Title: Density estimation using real nvp Year: (2017)
Ref_id:b10 Title: Towards understanding the invertibility of convolutional neural networks Year: (2017)
Ref_id:b11 Title: Deepmih: Deep invertible network for multiple image hiding Year: (2022-01)
Ref_id:b12 Title: Generating steganographic images via adversarial training Year: (2017)
Ref_id:b13 Title: Densely connected convolutional networks Year: (2017)
Ref_id:b14 Title: Steganography for health care and military applications Year: (2023)
Ref_id:b15 Title: Hinet: Deep image hiding by invertible network Year: (2021)
Ref_id:b16 Title: Securing data in internet of things (iot) using cryptography and steganography techniques Year: (2020-01)
Ref_id:b17 Title: Glow: Generative flow with invertible 1x1 convolutions Year: (2018)
Ref_id:b18 Title: Fixed neural network steganography: Train the images, not the network Year: (2022)
Ref_id:b19 Title: Representational aspects of depth and conditioning in normalizing flows Year: (2021)
Ref_id:b20 Title: Deep multiimage steganography with private keys Year: (2021-08)
Ref_id:b21 Title: High-capacity coverless image steganographic scheme based on image synthesis Year: (2023-02)
Ref_id:b22 Title: Large-capacity image steganography based on invertible neural networks Year: (2021)
Ref_id:b23 Title: Large-capacity and flexible video steganography via invertible neural network Year: (2023)
Ref_id:b24 Title: Secret communication using multi-image steganography for military purposes Year: (2022-07)
Ref_id:b25 Title: U-net: Convolutional networks for biomedical image segmentation Year: (2015-10)
Ref_id:b26 Title: Facenet: A unified embedding for face recognition and clustering Year: (2015)
Ref_id:b27 Title: Data hiding with deep learning: A survey unifying digital watermarking and steganography Year: (2023-12)
Ref_id:b28 Title: A conditional normalizing flow for accelerated multi-coil mr imaging Year: (2023)
Ref_id:b29 Title: Lightweight and effective deep image steganalysis network Year: (2022-08)
Ref_id:b30 Title: Mantra-net: Manipulation tracing network for detection and localization of image forgeries with anomalous features Year: (2019-06)
Ref_id:b31 Title: Generative flows with matrix exponential Year: (2020)
Ref_id:b32 Title: Invertible rescaling network and its extensions Year: (2023-01)
Ref_id:b33 Title: Robust invertible image steganography Year: (2022)
Ref_id:b34 Title: Towards universal training-free coverless image steganography with diffusion models Year: (2024)
Ref_id:b35 Title: Diffusion model makes controllable, robust and secure image steganography Year: (2024)
Ref_id:b36 Title: Universal deep hiding for steganography, watermarking, and light field messaging Year: (2020)
Ref_id:b37 Title: Depth-wise separable convolutions and multi-level pooling for an efficient spatial cnn-based steganalysis Year: (2020-08)
Ref_id:b38 Title: Versatile image watermarking for tamper localization and copyright protection Year: (2024)
Ref_id:b39 Title: Individualized image steganography method with dynamic separable key and adaptive redundancy anchor Year: (2024-11)
