Title: Learning the RoPEs: Better 2D and 3D Position Encodings with STRING
Abstract: Figure 1. Top: Successful diffusion policy conditioned on a STRING-enhanced Transformer vision encoder, attempting the doubleinsertion task on Aloha-sim. Bottom: Same experiment, but with a regular vision encoder for which the policy fails. STRING provides strong improvements for training dexterous robotics policies, outperforming previous position encoding algorithms such as RoPE.

Section: Introduction and Related Work
Position encodings (PEs) (Zhao et al., 2023;Kazemnejad et al., 2023;Chen et al., 2021;Kiyono et al., 2021) inject information about the respective locations of tokens into transformers (Vaswani et al., 2017). They are essential for good performance because vanilla attention is a set function, equivariant under permutation. In contrast, the meaning of a sequence of tokens in general depends on its ordering.
this section cite: ['b24', 'b5', 'b49']

Section: APEs and RPEs. Practitioners initially relied on absolute
PEs (APEs; Vaswani et al., 2017;Kiyono et al., 2021;Wang et al., 2021;Liu et al., 2020) which add or concatenate fixed, precomputed position embeddings to tokens. These have since been replaced by relative PEs (RPEs; Shaw et al., 2018;Raffel et al., 2020;Li et al., 2024;Chi et al., 2022;Press et al., 2022;Chi et al., 2023), which add a learnable bias term that depends on the distance between tokens to the pre-softmax attention logits. RPEs tend to generalise better than APEs over varying sequence lengths. However, they often require explicit computation for every query-key pair.
RoPE. To address the limitations of RPEs and APEs, researchers recently introduced rotary position encodings (RoPE; Su et al., 2024;Heo et al., 2025). These have been widely adopted in large language models (LLMs; Dubey et al., 2024;Gemma Team et al., 2024). RoPE acts on queries and keys by partitioning them into 2-dimensional blocks, each of which is rotated by an angle proportional to the token's position in the sequence. Whilst queries and keys are rotated separately, the angle of relative rotation is proportional to their separation, combining the best properties of APEs and RPEs. Mathematically, for query and key of dimensionality d, RoPE involves d 2 Givens rotations (Bindel et al., 2002) acting on disjoint 2D subspaces.
Besides providing strong empirical gains, two attractive properties have driven the enthusiastic uptake of RoPE. 1. Separability. RoPE transforms each query and key independently, based on its position. This happens once per token; the PE'd tokens are not recalculated during subsequent processing like autoregressive generation. This makes KV-caching convenient. Separability also makes RoPE compatible with linear attention, e.g. Performers (Choromanski et al., 2020;Katharopoulos et al., 2020).
Here, the attention matrix is not instantiated in memory so explicit RPE mechanisms are not possible.foot_0 2. Translational invariance. For a query-key pair at positions (i, j) ∈ N 2 , the relative rotation angle depends only on i -j. This improves sequence-length generalization.
However, RoPE is not the only position encoding algorithm with these desirable traits. In this paper, we propose a more general algorithm called STRING: Separable Translationally Invariant Position Encodings. STRING is based on Lie groups. It generalises RoPE via a unifying theoretical framework, incorporating the latter as a special case.
In fact, we later prove that STRING is the most general PE algorithm with the properties above, amongst a broad class.
STRING for robotics. The above features are especially important in robotics, where efficient 2D/3D token representation and sensible physical priors are key. To demonstrate it, we integrate STRING into Vision Transformers (ViTs), showing strong improvements for open-vocabulary object detection models and various robotics controllers. This showcases the real-world impact of our STRING.
Videos of STRING-based robotics controllers can be found here: https://sites.google.com/view/string-robotics.
this section cite: ['b49', 'b50', 'b32', 'b43', 'b38', 'b29', 'b7', 'b36', 'b8', 'b45', 'b21', 'b16', 'b17', 'b2', 'b9', 'b23']

Section: Key contributions.
1. We introduce STRING, a new family of position encodings for multidimensional token coordinates that respect both separability and translational invariance. 2. We rigorously analyse STRING's theoretical properties (Sec. 3), proving that it is more general than RoPE. We provide computationally efficient implementations. 3. We show strong accuracy gains across varied models using Transformers with STRING, on a range of robotics and general vision tasks (see Fig. 1 and Sec. 4).
this section cite: []

Section: Preliminaries
Let {x i } N i=1 ∈ R d denote a set of N d-dimensional tokens. Assume that d is even. The ith query, key and value vectors are given by q i = W q x i , k i = W k x i and v i = W v x i respectively, where W q , W k , W v ∈ R d×d are learned projection matrices (to keep the notation simple, we assume here the one-head setting). The attention mechanism, basic computational unit of the Transformer, can be written as:
x i → j exp(q i k j )v j l exp(q i k l ) .(1)
This updates the set of tokens, mixing them dynamically depending on the query-key softmax similarity scores.
Rotary position encodings. As described in Section 1, RoPE rotates tokens depending on their locations. In the 1D data setting (e.g. text), for a token z i ∈ {q i , k i } at position i ∈ N, we take z i → RoPE(i)z i with
RoPE(i)z i := d/2 n=1 ρ(iθ n )[z i ] 2n-1:2n ,(2)
ρ(θ) := cos θ -sin θ sin θ cos θ .(3)
Here, denotes the direct product, so that each 2×2 matrix {ρ(iθ n )} d/2 i=1 independently rotates a 2-element section of the query/key, and [z i ] 2n-1:2n denotes the 2n -1 and 2n elements of z i . Note that the matrix RoPE(i) is d × d, but it is only nonzero on the 2 × 2 blocks on the diagonal. Since ρ(θ) = ρ(-θ) and 2D rotations commute, we have that
RoPE(i) RoPE(j) = RoPE(j -i),(4)
whereupon we are transforming q i k j → q i RoPE(j-i)k j .
The dependence on j -i makes this translationally invariant. RoPE takes the set of angles {θ n } d/2 n=1 , determining the rotation frequency of each 2 × 2 block, as hyperparameters. We suppress this dependence for notational compactness. The authors originally proposed the decaying sequence θ n = λ -2(n-1)/d with base wavelength λ = 10, 000, though variants have since been explored (see below).
RoPE in higher dimensions. Whilst RoPE was originally proposed for sequence data, recent work has extended it to encode higher-dimensional position information (Heo et al., 2025). Now each token is equipped with a vector r i ∈ R dc , and we require: RoPE(r i ) RoPE(r j ) = RoPE(r j -r i ). Since 2D rotations commute, one approach is to define
RoPE(r i ) := dc k=1 RoPE([r i ] k ),(5)
where [r i ] k is the kth coordinate of r i (with k ∈ {1, ..., d c }). This independently applies regular 1-dimensional RoPE (Eq. 2) for each dimension of the position vector. The rotation frequencies {θ n } d/2 n=1 can optionally differ between each coordinate axis.
this section cite: ['b21']

Section: Generalisations of RoPE.
Prompted by its success, a number of papers have since sought to understand the effectiveness of RoPE and propose better-performing alternatives. One well-known method argues to increase the base wavelength λ to 500, 000, slowing the rate of token rotation and improving learning with longer contexts (Xiong et al., 2023;Roziere et al., 2023). Another suggests to completely truncate the lowest frequencies, setting them to zero, which helps preserve long-range 'semantic channels' (Barbero et al., 2024). Practitioners can also make the parameters {θ n } d/2 n=1 fully learnable, improving flexibility. Lastly, recent work has proposed to replace the block-diagonal RoPE matrices RoPE(i) by more general dense matrices in SO(d), parameterized by learned antisymmetric generators (Ostmeier et al., 2024). Whilst more expressive, this algorithm breaks translational invariance for position vectors with d c > 1, and has a large memory footprint. This makes it unsuitable for robotics applications. In Section 3, we will propose a better alternative, STRING.
this section cite: ['b51', 'b40', 'b0', 'b35']

Section: STRING: Separable Translationally Invariant Position Encodings
Recall that our goal is modify queries and keys depending on their respective positions, so that changes to dot products q i k j depend on r i -r j . RoPE achieves this using matrix multiplication (Su et al., 2024). Here, we present STRING: a more general, better-performing algorithm.
this section cite: ['b45']

Section: Introducing STRING
STRING is defined as follows.
Definition
3.1. STRING is the mapping R(•) : R dc → R d×d , from d c -dimensional position vec- tors to d × d matrices, given by R(r i ) = exp dc k=1 L k [r i ] k ,(6)
where {L k } dc k=1 ⊂ R d×d is a set of learnable and commuting skew-symmetric generators. Given a set of queries or keys
{z i } N i=1 ⊂ R d at positions {r i } N i=1 ⊂ R dc
, their positions are encoded as:
z i → R(r i )z i ∀i ∈ {1, ..., N }.(7)
Here, exp(•) refers to the matrix exponential, defined by its series expansion exp(A) := ∞ i=0 A i /i! and [r i ] k is the kth coordinate of vector r i . By 'commuting skewsymmetric generators', we mean that {L k } dc k=1 satisfy
[L i , L j ] = 0 and L i = -L i ∀ i, j.(8)
There are many ways to parameterize such a set; we give examples in Section 3.2. Remarkably, the following is true.
this section cite: []

Section: Theorem 3.2 (STRING is general).
Consider the set of mappings R(•) : R dc → R d×d that satisfy the grouplike translational invariance property R(r i ) R(r j ) = R(r j -r i ) ∀ r i , r j ∈ R dc , are continuously differentiable with respect to r i , and satisfy R(0) = I d (with I d the ddimensional identity). All such mappings can be expressed as STRING with some set of generators {L k } dc k=1 ⊂ R d×d .
In this sense, STRING is the most general of all translationally invariant position encoding mechanisms using matrix multiplication. Meanwhile, RoPE is a simple special case of STRING, taking a particular choice of generators. This can be seen as follows.
Theorem 3.3 (RoPE is a type of STRING #1). Consider the generators
L k = d/2 p=1 (δ 2p,2p-1 -δ 2p-1,2p
)θ p , where {θ p } d/2 p=1 ⊂ R and δ i,j is the delta function. This corresponds to RoPE with rotational frequencies {θ p } d/2 p=1 .
Proofs of Theorem 3.2 and Theorem 3.3 are in Appendix A.
this section cite: []

Section: Computationally efficient STRING
Despite being general and notationally compact, the parameterization of the STRING matrices R(r i ) shown in Eq. 6 may not be convenient for practical applications. Given N tokens at positions {r i } N i=1 , one must in general exponentiate and store N dense d × d matrices. This incurs O(N d 3 ) time complexity and O(N d 2 ) space complexity cost. The problem is exacerbated if the {r i } N i=1 differ across training examples and batches, which occurs e.g for point cloud data or color plus depth channel (RGB-D) images. In this case, {R(r i )} N i=1 cannot be cached and reused. This motivates the goal of this section: to find efficient STRING instantiations, nonetheless more general and expressive than RoPE. We begin with the following (proof in Appendix A): Theorem 3.4 (RoPE is a type of STRING #2). For any STRING position encoding with generators {L k } dc k=1 , there exists an orthogonal matrix P such that
R(r i ) = PRoPE(r i )P .(9)
Note that the orthogonal matrix P is independent of the coordinates r i , so it can be learned and stored once per attention head and shared across all training examples. Meanwhile, RoPE(r i ) is sparse -it is only nonzero on the super-and subdiagonals -so multiplying tokens only requires O(N d) memory and O(N dfoot_2 ) time, saving a factor of d. This is crucial in the contrastive learning setting where batch sizes can become large. Once again, one can see that RoPE is a special case of STRING, this time taking P = I d . We emphasize that the parameterization of STRING in Eq. 9 remains just as general as in Eq. 6. We also note that, since in regular attention one takes dot products between positionencoded queries and keys, the first orthogonal matrix P will always cancel with its counterpart. Therefore, in Transformers it is sufficient to take R(r i ) = RoPE(r i )P, with P ∈ O(d) learnable, without loss of generality. 2   Example 1: Cayley-STRING.
Equipped with Theorem 3.4, our goal becomes to choose a suitable parameterization for the orthogonal matrix P. One option is to take the Cayley Transform,
P Cayley := (I d -S)(I d + S) -1 , (10
)
where S is a learnable (potentially sparse) antisymmetric matrix (Diele et al., 1998). P Cayley is convenient since, for a token z i , we can compute (I d + S) -1 z i efficiently using a linear solver, avoiding expensive methods such as matrix inversions and matrix exponentials. Note that wherever we use P Cayley , we refer to our algorithm as Cayley-STRING.
The unreasonable effectiveness of STRING. In some sense, Theorem 3.4 makes it surprising that STRING outperforms RoPE so comprehensively in all our experiments (see Section 4), given that they are related by a change of basis. It appears that the ability to explicitly learn this basis change via P (shared between queries and keys), rather than implicitly via existing network weights, substantially boosts performance. Conversely, when using linear attention variants, the projected tokens W q q i and W k k i are pushed through nonlinearities such as ReLU(•) before taking the dot product. Hence, in this case, including learnable P does increase the capacity of the network, rather than simply learning a basis change.
Example 2: Circulant-STRING. We now present a second efficient STRING algorithm within our framework. A square matrix is referred to as circulant if it takes the form
C =         c 0 c d-1 • • • c 2 c 1 c 1 c 0 c d-1 • • • c 2 . . . c 1 c 0 . . . . . . c d-2 . . . . . . . . . c d-1 c d-1 c d-2 • • • c 1 c 0         .(11)
All rows are composed of the same elements, and each row is rotated one element relative to the preceding row. The transpose of a circulant matrix C is also circulant, and the sum of two circulant matrices is also circulant. It follows that C-C is circulant and antisymmetric. Lastly, circulant matrices commute. With these properties in mind, we can simply define L k = C k -C k for k ∈ {1, ..., d c }, with C k a learnable circulant matrix parameterized by d scalars {c 0 , ..., c d-1 }. We call this Circulant-STRING. This special parameterization is convenient for the following reason.
Theorem 3.5 (Circulant-STRING is fast). Given generators L k = C k -C k with C k circulant, the position encoding exp( L k [r i ] k )z i for token z i at position r i can be computed in O(d log d) time and O(d) memory using the fast Fourier Transform (FFT).
We provide a proof in Appendix A. Circulant-STRING provides another efficient position encoding algorithm that scales gracefully to large, high-dimensional datasets and performs well in spatial applications (see Section 4).
Learnable frequencies with STRING. Note that the STRING generators from Definition 3.1 are (in general) learnable. For Cayley-STRING, the angle-defining frequencies for RoPE and S, the skew-symmetric matrix from Equation (10) are learned whereas in Circulant-STRING, the scalars {c 0 , ..., c d-1 } in Equation ( 11) are learned.
STRING Train Inference Space Time Space Time Cayley O d 2 O d 3 O (d) Circulant O (d) O (d log d) O (d) O (d log d)
Table 1. Space and time complexity of the presented STRING methods in terms of the token dimensionality per head d. Computational complexity. Table 1 lists the computational complexity of the presented STRING methods. With token dimensionality per head d, Cayley-STRING introduces d 2 /2 parameters per head (from the d/2 angledefining RoPE frequencies and d (d -1) /2 parameters which fully determine the learnable skew-symmetric matrix used to generate P Cayley ). Circulant-STRING introduces d parameters per head (from the learnable circulant matrix fully determined by d parameters). In our experiments, d = 64 resulted in a negligible increase of trainable parameters. During training, Cayley-STRING takes O d 3 time due to the linear solver. However, during inference, the learned orthogonal matrix P Cayley can be absorbed into existing q/k projections at no extra cost (for vanilla attention). Circulant-STRING only takes O (d log d) time via the Fast Fourier Transform.
Both approaches improve upon RoPE, yet Cayley-STRING in general leads to larger improvements (see Section 4). Thus, we have here a classic trade-off between quality and computational expense. For applications with strict training performance constraints, we recommend Circulant-STRING due to its compact computational footprint, whereas for other applications Cayley-STRING is recommended.
this section cite: ['b13']

Section: Loose ends
Here, we discuss further generalisations of STRING.
Extension 1: ⊗-STRING. So far, we have followed RoPE in assuming that our position encodings are applied via matrix multiplication. However, this can be relaxed whilst preserving separability and translational invariance. For example, one can transform tokens z i via the outer product with position feature vectors f (r i ) ∈ R 2m ,
z i → vec(f (r i ) ⊗ z i ).(12)
Here, ⊗ denotes the outer product and vec denotes the 'vectorizing' operation that flattens a matrix to a vector, so that vec(f (r i ) ⊗ q i ) da+b = f (r i ) a q ib where a ∈ {1, ..., 2m} and b ∈ {1, ..., d}. Since the dot product of (flattened) outer products gives the product of dot products, we have
vec(f (r i ) ⊗ q i ) vec(f (r j ) ⊗ k j ) = q i k j • f (r i ) f (r j ).
(13) Now suppose that we take the Fourier features
f (r i ) = 1 √ m cos(ω k r i ), sin(ω k r i ) m k=1 ,(14)
where
{ω k } m k=1 ⊂ R d are learnable d-dimensional fre- quency vectors. Then we have that f (r i ) f (r r ) = 1 m m k=1 cos(ω k (r i -r j )) which is clearly a function of r i -r j .
We refer to this novel position encoding variant, orthogonal to previous RoPE-like approaches, as ⊗-STRING.
Extension 2: General transformation groups. Having focused on translational invariance, another natural question is whether STRING could be repurposed for other continuous transformation groups. These may be more suitable for data with different properties; for example, one might sometimes prefer a rotationally invariant position encoding.
More formally, recall that a Lie group with parameters ψ ∈ R k is a group of transformations of the form T ψ : R d → R d that are differentiable with respect to ψ. Let the parameter ψ = 0 correspond to the identity element, so that T 0 x = x. A canonical coordinate system for G is an injective map ρ from Cartesian coordinates to a new coordinate system, satisfying ρ(T ψ x) = ρ(x) + k i=1 ψ i e i ∀ T ψ ∈ G, where e i is the ith standard basis vector. Observe that the right hand side of this equation represents a translation in the new basis. Canonical coordinate systems exist for all one-parameter Lie groups (k = 1), and more generally for Abelian groups of dimension k ≤ d (Segman et al., 1992;Rubinstein et al., 1991;Tai et al., 2019). They can be derived analytically by solving a set of first-order PDEs, though for many common transformation groups the canonical coordinate system is obvious. For instance, for azimuthal and polar rotations of points (r x , r y , r z ) in 3D space (k = 2), a canonical coordinate system is (θ, φ), where sin θ := r 2
x + r 2 y / r 2 and tan φ := r y /r x . Rotatingfoot_3 simply 'translates' the canonical coordinates (θ, φ) → (θ + ∆θ, φ + ∆φ) -a transformation looking much more complicated in the Cartesian basis.
STRING for Abelian Lie groups. It follows that, simply by replacing Cartesian coordinates {r i } N i=1 with their canonical counterparts, we can repurpose STRING to construct position encodings that respect more general invariances.
this section cite: ['b42', 'b41', 'b46']

Section: Experiments
In this section, we provide an exhaustive empirical comparison of STRING with RoPE and vision encoders leveraging regular APEs. To set the stage, we start with general nonrobotics experiments in Sec. 4.1. On our way to robotics applications, we then test STRING for 2D and 3D object detection in Sec. 4.2. Finally, we present robotics manipulation experiments in Sec. 4.3 and Sec. 4.4.
this section cite: []

Section: General Experiments: Classification and Retrieval
We tested STRING for image classification tasks on the ImageNet2012 (Deng et al., 2009) and Places365 datasets, with Vision Transformer (ViT) (Dosovitskiy et al., 2021) as our base model. We compare against RoPE and RoPE-Mixed (Heo et al., 2025), abbreviated to RoPE-M, to Circulant-STRING and Cayley-STRING (respectively abbreviated to Circulant-S and Cayley-S). The results are shown in Table 2. For both datasets, STRING offers best models. For ImageNet2012, the top two models are STRINGs. Furthermore, ImageNet2012 STRINGs provide absolute gains larger than 1%, as compared to regular ViTs, with only a negligible set of extra trainable parameters. Table 4. Average Precision (AP) % of the OWL-ViT model on COCO (Lin et al., 2014) and LVIS (Gupta et al., 2019). Best in bold, second-best underlined.
Anything-V2 (Yang et al., 2024) for metric mono-depth estimation. The dataset is filtered using methods from (Chen et al., 2024) to remove images on which the indoor-finetuned model performs poorly such as those with overlays, no visible groundplane, large outdoor scenes, or optical illusions.
We perform contrastive learning on the text to visual representation pairs in the WebLI-3D lifted dataset, where the visual representation may be in the form of an RGB image or an RGB-D depth image. Similar to CLIP (Radford et al., 2021), this is done by maximizing the similarity between the embeddings of matching visual-text pairs, while minimizing the similarity between embeddings of the non-matching pairs. This enables open-vocabulary detection and classification by comparing the text embeddings of all possible classes against those of the visual representation, and selecting the minimum distance pair. We compare against baseline in Table 3. For all six evaluations, Cayley-STRING is the best and Circulant-STRING is second best.   Baseline RoPE Cayley-STRING ViT ViTD Figure 2. Example outputs for the 3D detection task for baseline, RoPE, and Cayley-S. Green boxes: groundtruth. Blue boxes: predictions.
our policies on real ALOHA 2 robots (see Fig. 3 and Fig. 11). Due to the large observed variance of the on-robot evaluation for ALOHA 2, we focused on the evaluation in simulation to accurately rank different methods.
Table 6 reports the best task success rate of using RoPE (Heo et al., 2025), and Cayley-STRING on a baseline SigLIP B/16 256 (Zhai et al., 2023) ViT model. The success rate is averaged over 10 trials of each checkpoint, taken every 10K train steps and over 1M train steps. Corresponding curves are given in Fig. 4. Cayley-STRING achieves superior results across all tasks on average (i.e. MultiTask). Additionally, it achieves equivalent or superior results, as compared to RoPE (e.g. for the DoubleInsertion task from Fig. 1) and ViT for all 12 tasks except for Mu-gOnPlate and PlateOnRack (second-best). Finally, STRING converges much faster than other methods (see: Fig. 4). Note that we applied the strategy of learning all angledefining frequencies for both RoPE and Cayley-STRING. Figure 4. Mean success rate across all tasks (i.e. MultiTask) evaluated 10 times every 10K train steps over 1M train steps. ViT RoPE STRING BowlOnRack 0.90 0.80 1.00 DoubleInsertion 0.20 0.50 0.60 FMB-1 0.20 0.20 0.20 FMB-2 0.10 0.10 0.10 FruitBowl 0.30 0.30 0.30 GlassOnRack 0.60 0.60 0.60 HandOverBanana 1.00 1.00 1.00 HandOverPen 1.00 1.00 1.00 MugOnPlate 0.70 0.90 0.80 PlateOnRack 0.60 0.70 0.50 SingleInsertion 0.40 0.60 0.60 StorageBin 0.00 0.00 0.00 MultiTask 0.37 0.42 0.46 Table 6. Mean success rate (best in bold, second-best underlined) over 10 evaluations of each ALOHA simulation task. RoPE (Heo et al., 2025) and Cayley-STRING are added to a baseline SigLIP B/16 256 ViT (Zhai et al., 2023). See Appendix B for details.
this section cite: ['b12', 'b14', 'b21', 'b30', 'b18', 'b52', 'b4', 'b37']

Section: Real-World 3D Robot Manipulation: KUKA
Establishing STRING as superior to other methods on previous tasks, we let it operate on 3D data to obtain new SOTA robotic manipulation policies. This resulted in policies directly using depth and deployed on real hardware. Note that STRING can be naturally applied in that context since it can be used for data equipped with general coordinate vectors r i associated with tokens (e.g. 3D).
this section cite: []

Section: SETTING
We evaluated STRING in the vision encoder of a generative policy applying energy-based models (Singh et al., 2024) and deployed on a real industrial KUKA robot arm (Udayan et al., 2023). The closed-loop feedback policy operates on RGB-D images, and is learned as a generative model with imitation learning. Its architecture (see Appendix G for details) consists of a diffusion Transformer and 3D encoders. The policy was trained on a mixture of scripted and teleoperated data collected for 3 different skills (pick, place and handover) on various objects. It is evaluated exclusively on the pick skill with a diverse set of objects. Each evaluation was run as an A/B test for a total of 50 trials.
this section cite: ['b44', 'b48']

Section: REGULAR EVALUATIONS
We experimented with two ways of using depth in the policy.
Implicit depth via normal maps: In the first approach, following (Tziafas & Kasaei, 2023), depth input is used to construct a surface normal map with unit R 3 values per pixel. Both RGB and depth inputs are then processed via identical (shared weights) embedding layers. The embeddings are concatenated and processed through Transformer layers.
Finally, the embeddings are split and fused to yield the final vision embedding. Our results in Figure 5 show that this approach of incorporating depth has a detrimental effect on the on-robot deployed policy. We hypothesize that this is caused by the significant amount of noise coming from the depth sensors, leading to imperfect surface normal maps.
Lifting patches to 3D for STRING: In the second approach, we compute the height for each patch via meanpooling across depth values for all the pixels in the patch, followed by the learnable linear layer. The resulting 3D patches are then fed to the Transformer, with positional encoding given by STRING to incorporate depth into the vision encoder. Our results Figure 5 show that STRING improves the success rate over the 2D base policy. Also, when STRING is combined with the first method, it drastically reduces the negative impact of noisy normal maps.
We used Circulant-STRING to obtain a particularly compact computational footprint. Note that in this setting, more computationally intense mechanisms, such as (Ostmeier et al., 2024), ran out of memory and could not be trained.
this section cite: ['b47', 'b35']

Section: OUT-OF-DISTRIBUTION EVALUATION: STRING VS BASELINE
To further compare STRING with the baseline and show the advantages of using 3D over 2D policies, we also perform out-of-distribution (OOD) evaluations on the real robot.
We vary three different environment settings. These include:
(1) lighting changes, (2) adding large distractor objects and
(3) changing the height of the table from which the robot has to grasp the block. For each setting, we test multiple different variations, e.g., three different light settings. 65.3 42.2 53.1 73.8 0 20 40 60 80 2D 2D + nmap nmap + 3D STRING 3D STRING Success Rate Figure 5. Performance of STRING with 3D input vs. baselines on real-robot tasks (with 2 seeds). 2D baseline performance without depth input is ≈ 65%.
Incorporating depth through surface normal maps (nmap) reduces performance to 42%. Using 3D STRING for incorporating depth improves the performance in both scenarios -with and without normal maps to 53% and 74% respectively. Mean/stdev shown above were calculated from 35 evaluation runs.
policies from Section 4.4.2. As seen in Figure 6, 3D STRING policies outperform 2D policies across all OOD settings. For instance, with large distractors (middle), the 2D model's performance decreases from 65% to 57%, while 3D STRING maintains performance similar to non-OOD settings (≈ 74%). In some OOD cases, such as lighting changes, both 2D (≈ 10%) and 3D (≈ 25%) policies experience a performance decrease vs. the non-OOD setup. This drop in performance during lighting changes is likely due to the significant alteration in image observations, thus affecting both 2D and 3D policies. Finally, the largest performance difference is observed in the table height variation scenario. Here, the 3D policies exhibit significantly higher robustness (≈ 50%) compared to the 2D policies (≈ 10%). This suggests that the 3D STRING policy leverages the raw depth signal to better generalize to table height variations, a change imperceptible to fixed monocular cameras.
Overall, our results show that 3D STRING policies are highly robust to many variations and significantly improve over 2D policies. Fig. 7 shows a sample episode from the on-robot evaluation of the STRING generative policy.
this section cite: []

Section: Lighting Large Distractors
Table Height  From 2D to 3D with STRING: We have already demonstrated (see the normal map approach from Section 4.4.2) that just adding a depth signal does not necessarily improve performance. STRING does so and exhibits another feature that other approaches (e.g. adding depth as extra channel) do not: it can be trained from a regular 2D pre-trained checkpoint. This is the case since STRING incorporates depth by using it to modulate a regular attention matrix, effectively disentangling 3D specific parameters (defining the modulation) from the core 2D backbone. All training runs in Section 4.4 started from the pre-trained 2D backbones.
this section cite: []

Section: Conclusion
We introduced a new class of translationally invariant position encodings (PEs) for Transformers, called STRING. STRING is the most general of all translation-invariant PE methods using matrix multiplications (under weak smoothness assumptions) and contains the prominent class of RoPE methods as its special instantiation. We proposed to apply STRING in robotics for 2D and 3D modeling and provided its extensive empirical verification over a range of tasks, from standard classification and retrieval, through object localization, to diffusion robotic policies conditioned on Vision Transformers. In all these experiments, we showed consistent gains over RoPE, as well as baselines applying regular absolute position encodings.
this section cite: []

Section: References
Ref_id:b0 Title: Round and round we go! what makes rotary positional encodings useful? arXiv preprint Year: (2024)
Ref_id:b1 Title: A versatile 3B VLM for transfer Year: (2024)
Ref_id:b2 Title: On computing givens rotations reliably and efficiently Year: (2002)
Ref_id:b3 Title: Benchmarking in manipulation research: The ycb object and model set and benchmarking protocols Year: (2015)
Ref_id:b4 Title: Endowing vision-language models with spatial reasoning capabilities Year: (2024)
Ref_id:b5 Title: A simple and effective positional encoding for transformers Year: (2021-11-11)
Ref_id:b6 Title: A jointlyscaled multilingual language-image model Year: (2023)
Ref_id:b7 Title: KER-PLE: kernelized relative positional embedding for length extrapolation Year: (2022-11-28)
Ref_id:b8 Title: Dissecting transformer length extrapolation via the lens of receptive field analysis Year: (2023)
Ref_id:b9 Title: Rethinking attention with performers Year: (2020)
Ref_id:b10 Title: From block-toeplitz matrices to differential equations on graphs: towards a general theory for scalable masked transformers Year: (2022)
Ref_id:b11 Title: Dataset and benchmarks for real-world 3d object understanding Year: (2022)
Ref_id:b12 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b13 Title: The cayley transform in the numerical solution of unitary differential systems Year: (1998)
Ref_id:b14 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b15 Title: Google scanned objects: A high-quality dataset of 3d scanned household items Year: (2022)
Ref_id:b16 Title: The llama 3 herd of models Year: (2024)
Ref_id:b17 Title: Open models based on gemini research and technology Year: (2024)
Ref_id:b18 Title: Lvis: A dataset for large vocabulary instance segmentation Year: (2019)
Ref_id:b19 Title: Lie groups, Lie algebras, and representations Year: (2013)
Ref_id:b20 Title: Bridging nonlinearities and stochastic regularizers with gaussian error linear units Year: (2016)
Ref_id:b21 Title: Rotary position embedding for vision transformer Year: (2025)
Ref_id:b22 Title: Integrating generic sensor fusion algorithms with sound state representations through encapsulation of manifolds Year: (2011)
Ref_id:b23 Title: Transformers are rnns: Fast autoregressive transformers with linear attention Year: (2020)
Ref_id:b24 Title: The impact of positional encoding on length generalization in transformers Year: (2023)
Ref_id:b25 Title: SHAPE: Shifted absolute position embedding for transformers Year: ()
Ref_id:b26 Title: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing Year: (2021-11)
Ref_id:b27 Title: URL Year: ()
Ref_id:b28 Title: The hungarian method for the assignment problem Year: (1955)
Ref_id:b29 Title: Functional interpolation for relative positions improves long context transformers Year: (2024)
Ref_id:b30 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b31 Title: Proc4gem: Foundation models for physical agency through procedural generation Year: (2025)
Ref_id:b32 Title: Learning to encode position for transformer with continuous dynamical model Year: (2020-07-18)
Ref_id:b33 Title: fast and accurate: Kernelized attention with relative positional encoding Year: (2021)
Ref_id:b34 Title: Simple open-vocabulary object detection with vision transformers Year: (2022)
Ref_id:b35 Title: Generalizing rotary position encodings Year: (2024)
Ref_id:b36 Title: Train short, test long: Attention with linear biases enables input length extrapolation Year: (2022)
Ref_id:b37 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b38 Title: Exploring the limits of transfer learning with a unified textto-text transformer Year: (2020)
Ref_id:b39 Title: Linear transformer topological masking with graph random features Year: (2024)
Ref_id:b40 Title: Code llama: Open foundation models for code Year: (2023)
Ref_id:b41 Title: Recognition of distorted patterns by invariance kernels Year: (1991)
Ref_id:b42 Title: The canonical coordinates method for pattern deformation: Theoretical and computational considerations Year: (1992)
Ref_id:b43 Title: Self-attention with relative position representations Year: (2018)
Ref_id:b44 Title: Revisiting energy based models as policies: Ranking noise contrastive estimation and interpolating energy models Year: (2024)
Ref_id:b45 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b46 Title: Equivariant transformer networks Year: (2019)
Ref_id:b47 Title: Early or late fusion matters: Efficient rgb-d fusion in vision transformers for 3d object recognition Year: (2023)
Ref_id:b48 Title: Forward kinematics simulation of KUKA KR5 arc robot with robo analyzer Year: (2023)
Ref_id:b49 Title: Attention is all you need Year: (2017-12-04)
Ref_id:b50 Title: On position embeddings in BERT Year: (2021)
Ref_id:b51 Title: Effective long-context scaling of foundation models Year: (2023)
Ref_id:b52 Title: Depth anything v2 Year: (2024)
Ref_id:b53 Title: Sigmoid loss for language image pre-training Year: (2023)
Ref_id:b54 Title: Length extrapolation of transformers: A survey from the perspective of position encoding Year: ()
Ref_id:b55 Title: Aloha unleashed: A simple recipe for robot dexterity Year: (2024)
