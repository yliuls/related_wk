Title: Online Functional Tensor Decomposition via Continual Learning for Streaming Data Completion
Abstract: Online tensor decompositions are powerful and proven techniques that address the challenges in processing high-velocity streaming tensor data, such as traffic flow and weather system. The main aim of this work is to propose a novel online functional tensor decomposition (OFTD) framework, which represents a spatialtemporal continuous function using the CP tensor decomposition parameterized by coordinate-based implicit neural representations (INRs). The INRs allow for natural characterization of continually expanded streaming data by simply adding new coordinates into the network. Particularly, our method transforms the classical online tensor decomposition algorithm into a more dynamic continual learning paradigm of updating the INR weights to fit the new data without forgetting the previous tensor knowledge. To this end, we introduce a long-tail memory replay method that adapts to the local continuity property of INR. Extensive experiments for streaming tensor completion using traffic, weather, user-item, and video data verify the effectiveness of the OFTD approach for streaming data analysis. This endeavor serves as a pivotal inspiration for future research to connect classical online tensor tools with continual learning paradigms to better explore knowledge underlying streaming tensor data.

Section: Introduction
In real world, high-dimensional data often exist in a streaming form and are typically modeled as tensor streams (such as traffic flow and video) Yu et al. [2015], Smith et al. [2018]. Tensor streams can be categorized into two types: single-aspect streams and multi-aspect streams. Single-aspect streams, such as traffic flow data represented by the triplet (location, route, and time), grow only along the temporal dimension and are modeled as 3-mode temporal tensor streams. In contrast, multi-aspect streams, such as recommendation system data represented by the triplet (user, movie, and actor), grow along multiple dimensions simultaneously. With the increasing prevalence of streaming data, there is a growing demand for real-time streaming data analysis (e.g., streaming data completion).
Online tensor decomposition Abed-Meraim et al. [2022a] is one of the foremost methods to address the streaming data completion problem by exploiting the potential compact structures underlying streaming data within an online optimization framework. However, decomposing tensor streams would lead to high computational costs owing to the significant growth in their volume over time. Also, dynamically capturing the internal latent properties of tensor streams, such as spatial-temporal continuity, poses difficult challenges. To address these challenges, various online tensor decomposition methods have been developed to handle the streaming data completion problem. These The proposed online functional tensor decomposition paradigm for streaming tensor data. Our approach transforms the static online optimization algorithm into a more dynamic continual learning paradigm, which learns knowledge from streaming data by using the CP decomposition parameterized by INRs.
methods can be categorized into CP decomposition-based methods Lee and Shin [2021], Zhong et al. [2021], Ahn et al. [2021], Abed-Meraim et al. [2022b], Liu et al. [2022], Tucker decomposition-based methods Jang and Kang [2023], tensor SVD (t-SVD)-based methods Zhang et al. [2016], Wu [2022], Gilman et al. [2022], Yi et al. [2022], and other tensor network decomposition-based approaches (such as tensor train (TT) decomposition Le et al. [2024]). As recent symbolic works along this line, Ahn et al. [2021] proposed a streaming tensor factorization with an attention-based temporal regularization for streaming data completion. Abed-Meraim et al. [2022b] proposed a scalable online CP algorithm that efficiently estimates low-rank components from streaming tensors. Yi et al. [2022] proposed an online algorithm based on the t-SVD, which can efficiently capture the principal information of tensor by incrementally updating a much smaller sketch. These methods enable real-time analysis of tensor streams and achieve promising performances for streaming data completion. Nonetheless, current online tensor decomposition methods mostly consider optimizing the factor matrices/tensors of the tensor decomposition using discrete online optimization rules, which may limit the capability for dynamic data structure modeling and spatial-temporal correlation excavation.
In this work, we propose a novel online functional tensor decomposition (OFTD) method for streaming data completion (see Fig. 1(b) for quick view). OFTD represents the spatial-temporal streaming data as a continuous function using CP tensor decomposition parameterized by implicit neural representations (INRs) Sitzmann et al. [2020], Mildenhall et al. [2021], which map an arbitrary spatial-temporal coordinate to the corresponding tensor value through deep neural networks. Instead of optimizing the factor matrices of tensor decomposition, OFTD optimizes the learnable weights of factor INRs during online optimization, which could better explore the complex dynamic structure and global spatial-temporal correlations of streaming data through a deep functional tensor representation. OFTD allows for natural characterizations of continually expanded streaming data by simply adding new coordinates into the INR network during online optimization. Although the tensor stream is continually growing, we maintain a constant number of optimization parameters by keeping the network structure unchanged.
Nevertheless, the incorporation of deep neural networks leads to the possible issue of forgetting the learned historical knowledge when fitting new data. Hence, to enable effective continual learning for processing streaming tensors using OFTD, we make a basic attempt by employing the memory replay method, which utilizes a part of historical data during online optimization, to avoid forgetting. We develop a regret bound of OFTD from the perspective of the loss (i.e., forgetting) of historical knowledge, which theoretically shows that the OFTD model tends to forget the knowledge that is distant from the new tensor data due to the local continuity property of INRs. Hence we correspondingly design a memory buffer with a long-tail distribution (i.e., storing more data at more distant positions), which substantially enhances the performance for streaming data completion. Consequently, OFTD delivers superior performances against traditional online tensor decomposition methods, showcasing its strong ability for streaming data analysis. We summarize the main contributions of this work as follows:
• We introduce a novel online functional tensor decomposition (termed OFTD) method for streaming data completion. Our method employs the CP decomposition parameterized by INRs to learn a spatial-temporal continuous function, which enables a concise representation of streaming tensor data by simply incorporating new coordinates into the INR during online optimization. Furthermore, our method effectively captures spatial-temporal continuity and low-dimensional compact structures of streaming data through functional representation.
• Our approach transforms the static online algorithm for streaming data completion into a more dynamic continual learning paradigm, i.e., the INRs are expected to fit new data without forgetting previous tensor knowledge. To achieve this, we theoretically develop a regret bound for OFTD, which guides us in designing a long-tail replay continual learning method tailored for OFTD.
• We apply OFTD to single-aspect (i.e., temporal evolution) and multi-aspect (spatial and temporal evolutions) streaming tensor completion. Extensive experiments on real-world datasets show the superiority of OFTD over state-of-the-art online tensor decomposition methods.
2 Related Work
this section cite: ['b47', 'b39', 'b20', 'b50', 'b3', 'b23', 'b15', 'b48', 'b42', 'b12', 'b46', 'b3', 'b46', 'b28']

Section: Online Tensor Decomposition
Online analysis algorithms for streaming tensors have been widely developed in recent years Najafi et al. [2019], Qian et al. [2021], Hu et al. [2022]. CP decomposition is one of the mostly considered methods for streaming tensor completion. For instance, Minh-Chinh et al. [2016] proposed a twostage CP decomposition algorithm to perform streaming data completion for third-order tensors. Lee and Shin [2021] proposed a robust method for tensor streams by integrating CP factorization, outlier removal, and temporal-pattern detection to enable accurate online prediction. Zhong et al. [2021] proposed a window-based dynamic streaming tensor analysis method using CP decomposition. 3 Proposed Method
this section cite: ['b30', 'b14', 'b20', 'b50']

Section: Preliminaries
Basic notations are shown in Table 1. We use • to denote the Kruskal operator Kolda and Bader [2009]. The • and ⊗ respectively denote the outer product and the Hadamard product. The ∥•∥ F denotes the Frobenius norm.
Table 1: Notations used in this paper.
Notations Definitions x, x, X, X Scalar, vector, matrix, and tensor X t ∈ R I t
1 ×I t 2 ×•••×I t N
An N th -order streaming tensor X (i,:) , X (:,i)
The i-th row or the i-th column of X X (i1,i2,...,i N ) ∈ R The (i
1 , i 2 , . . . , i N )-th element of X U (n) ∈ R In×r
The n-th factor of CP decomposition [N ] ∈ Z N
The vector
[N ] ≜ (1, 2, • • • , N ) T
Definition 1 (CP Decomposition Kolda and Bader [2009]). Given an N th -order tensor X ∈
R I1×I2×•••×I N , its CP decomposition is the representation using N factor matrices U (n) N n=1
sharing the same number of columns as follows:
X = U (n) N n=1 ≜ r i=1 U (1) (:,i) • • • • • U (N ) (:,i) ,(1)
where the factor matrix U (n) ∈ R In×r . The smallest integer r satisfying (1) is referred to as the CP-rank of X .
this section cite: []

Section: Definition 2 (Streaming tensor sequence).
A sequence of N th -order tensors {X t } is called streaming tensor sequence if for any t ∈ Z + , X t ⊆ X t+1foot_0 . The t grows with time, and X t is called the snapshot tensor taken at time t. Definition 3 (Temporal tube). Given two successive tensors X t-1 ∈ R
I t-1 1 ×•••×I t-1 N and X t ∈ R I t 1 ×•••×I t N
derived from a streaming tensor sequence {X t }, the coming data (i.e., temporal tube) at time t can be represented by Y t = X t \ X t-1 , which has the same size as X t , with entries given by:
(Y t ) (i1,...,i N ) = (X t ) (i1,...,i N ) if ∃ I t-1 n < i n ≤ I t n , 0 otherwise.
this section cite: []

Section: Problem Formulation
Given a streaming tensor sequence {X t } with missing entries, we aim to recover the missing data in the current snapshot X t . Since X t-1 ⊆ X t , and we have handled X t-1 in previous time steps, the problem is equivalent to completing the elements in Y t = X t \ X t-1 . Tensor streams often exhibit low-rank properties, making tensor decomposition models, such as the CP decomposition, suitable for modeling streaming tensors. The optimization problem at time t is typically formulated as
min U (n) t P t ⊗ Y t - U (n) t N n=1 F + R U (n) t ,(2)
where
U (n) t
denotes the set of tensor factors, P t is a binary tensor representing missing and observed entries of Y t , and R(•) is a regularization term for the factor matrices. For instance, most tensor streams exhibit continuity (i.e., smoothness) over the spatial and time dimensions, e.g., the current pollutant measurement is similar to those of the previous and next 10 minutes. Hence, smooth regularizations such as the total variation Pragliola et al. [2023] can be employed. However, in this work we do not impose an explicit regularization R(•). Instead, the proposed OFTD implicitly captures the spatial-temporal smoothness of data through the implicit smoothness of INRs (see Lemma 1).
this section cite: ['b33']

Section: Online Functional Tensor Decomposition
In this section, we give the detailed formulation of the proposed online functional tensor decomposition for streaming data completion.
Functional Tensor Decomposition Before introducing the online algorithm, we first introduce the batch functional tensor decomposition (FTD) (i.e., off-line setting), which uses INRs to parameterize factor matrices of the CP decomposition. Specifically, for the n-th CP factor matrix U (n) t ∈ R I t n ×r , we use an INR Sitzmann et al. [2020] to parameterize it. Such INR is a multilayer perceptron (MLP) f Θn (•) : R → R r with parameters Θ n , which takes a coordinate v ∈ R as input and returns a vector:
f Θn (v) = W d (σ(W d-1 • • • σ(W 1 v))) ∈ R r ,(3)
where Θ n ≜ {W i } d i=1 are weight matrices with W i ∈ R r×r for i = 2, 3, . . . , d and W 1 ∈ R 1×r . σ(•) = sin(ω 0 •) is the sine activation function that is more effective for INR Sitzmann et al. [2020]. To generate the factor matrix U (n) t , we consider the parallel notation for I t n input coordinates: f Θn (v) ≜ f Θn (v (1) ), . . . , f Θn (v (I t n ) ) T ∈ R I t n ×r , where v = [I t n ] ≜ (1, 2, . . . , I t n ) T represents the input coordinate vector and v (i) denotes the i-th element of the vector v. We use this output matrix f Θn (v) (i.e., f Θn ([I t n ])) as the factor matrix U (n) t
of the CP decomposition and then obtain the FTD representation of an N th -order streaming tensor X t as
X t = f Θn ([I t n ]) N n=1 ≜ r i=1 f Θ1 ([I t 1 ]) (:,i) • • • • • f Θ N ([I t N ]) (:,i) ,(4)
where f Θn ([I t n ]) (:,i) denotes the i-th column of the factor matrix f Θn ([I t n ]). We have hence used N INRs {f Θn (•)} N n=1 to parameterize the factor matrices of the CP decomposition. The FTD in (4) enjoys two potential advantages for streaming data analysis. First, the functional representation naturally allows us to model streaming data in an efficient way by simply adding new coordinates into the model (i.e., by expanding the coordinates [I t n ]) during online optimization. With the expanding of coordinates over time, the output of INR (i.e., the factor matrix of CP decomposition) correspondingly increases in size to fit the larger size of the new data stream. Second, the FTD is effective for modeling real-world tensor data because the CP decomposition characterizes the intrinsic low-dimensional structure of the tensor, while the INRs capture the tensor spatial-temporal smoothness (see Lemma 1) to better recover the unobserved entries.
Online Functional Tensor Decomposition We now introduce the proposed OFTD. We mainly describe the multi-aspect setting, and the single-aspect problem can be seen as a special case of the multi-aspect problem. In the online setting, a new tensor X t ∈ R I t 1 ×I t 2 •••×I t N arrives at each time t with missing values. We dynamically update the FTD model to accommodate the incremental growth of the streaming tensor. Our streaming tensor completion algorithm consists of the initialization stage and the online update stage.
this section cite: []

Section: Initialization Stage: Given an initial streaming tensor X
1 ∈ R I 1 1 ×I 1 2 ×•••×I 1
N , we optimize the following objective function based on the FTD representation (4):
min {Θn} P 1 ⊗ Y 1 -f Θn ([I 1 n ]) N n=1 2 F ,(5)
where P 1 is the initial mask, Y 1 = X 1 is the initial observed tensor, Θ n are learnable parameters of the INR f Θn (•), and [I 1 n ] denotes the coordinates of the tensor f Θn ([I 1 n ]) N n=1 at dimension n. To address the optimization problem, we use gradient descent-based methods (e.g., the Adam optimizer) to update the INR parameters {Θ n } N n=1 with the loss (5).
this section cite: []

Section: Online Stage:
In the online update stage, a new tensor X t ∈ R I t 1 ×•••×I t N (t ≥ 2) comes at each time point and we need to update the online FTD model to accommodate the growth of streaming tensor sizes. We take the update of the n-th dimension as an example. At time t, the streaming tensor size grows by the scale of I t n -I t-1 n along the n-th dimension. We adapt to the new tensor size by simply adding I t n -I
t-1 n coordinates to the corresponding factor INR f Θn (•) : R → R r . Specifically, suppose that the coordinate vector at the last time point is [I t-1 n ] ≜ (1, • • • , I t-1 n ) T , then the new coordinate vector at the time point t is [I t n ] ≜ (1, • • • , I t n ) T . Correspondingly, the new factor matrix of the CP decomposition along the dimension n at the time point t is obtained by f Θn ( I t n ) ≜ f Θn (1), f Θn (2), . . . , f Θn (I t n ) T ∈ R I t n ×r . Historical Historical New Historical Historical New We illustrate the dynamic growth of the coordinates vector [I t n ] and the corresponding OFTD model in Fig. 1(b). Based on the enlarged OFTD model, we consider the following optimization problem during the online stage t: min {Θn} P t ⊗ Y t -f Θn ([I t n ]) N n=1 2 F . (6) Similarly, we use the Adam optimizer to update the INR parameters {Θ n } N i=1 with the loss (6) at each time point t. Here, the observed data Y t = (X t \ X t-1 ) ∈ R I t 1 ×•••×I t N contains the new data at time t, i.e., the INRs {f Θn (•)} N n=1 are optimized to fit the new data in X t \ X t-1 , with initialization weights {Θ n } being taken from the last time point (i.e., the continual learning). Such online optimization strategy is reasonable since it is impractical to use all historical data at each time point, as this would result in large computational costs. Therefore, we consider only fitting the new data to optimize the INRs in each time point t. For single-aspect streams, we continually expand the temporal coordinates of OFTD, while for multi-aspect streams, we need to continually expand the coordinates in all dimensions n = 1, • • • , N . Given a total time T , the completion result is obtained by the FTD model f Θn ([I T n ]) N n=1 ∈ R I T 1 ×•••×I T N after T optimization steps under the continual learning manner.
Notably, the streaming data completion using OFTD becomes a classical continual learning paradigm, which learns the INR weights {Θ n } from a continuous stream of information, with such information becoming progressively available over time Parisi et al. [2019]. Nevertheless, OFTD may encounter forgetting of historical data during online optimization, which we will analyze and address next.
this section cite: ['b32']

Section: Theoretical Analysis
We interpret two insights of our method, i.e., the spatial-temporal continuity of OFTD brought from INR, and the regret bound of the OFTD model that reveals its forgetting behavior, which motivates us to design memory replay to alleviate forgetting. First, we show that our OFTD method preserves the tensor spatial-temporal smoothness from the Lipschitz smooth perspective.
Lemma 1 (Lipschitz smooth bound for FTD). Let the tensor X t ∈ R I t
1 ×I t 2 ×I t 3 satisfy FTD (4), where each factor function f Θn (•) : R → R r (n = 1, 2, 3) is an INR formulated as in (3) with activation function σ(•) = sin(ω 0 •).
Assume that each element of the weight matrix W in (3) follows i.i.d. N (0, w 2 ). Then for any δ ∈ (0, 1), any spatial coordinates
v, v ′ ∈ Z 2 , where v = (v (1) , v (2) ), v ′ = (v ′ (1) , v ′ (2)
) and any temporal coordinates k, k ′ ∈ Z, with probability at least 1 -δ the following Lipschitz smoothness holds for X t :
|(X t ) (v (1) ,v (2) ,k) -(X t ) (v ′ (1) ,v ′ (2) ,k) | ≤ C 1 ∥v -v ′ ∥ l1 Spatial smooth , |(X t ) (v (1) ,v (2) ,k) -(X t ) (v (1) ,v (2) ,k ′ ) | ≤ C 1 |k -k ′ | Temporal smooth ,(7)
where
C 1 = ω 3d-3 0 (2wr 2 + w ln 3d δ ) 3d max(I t 1 I t 2 , I t 1 I t 3 , I t 2 I t 3 ) is a Lipschitz constant.
Lemma 1 shows that the FTD model preserves the spatial-temporal continuity of the estimated tensor X t (i.e., elements that are closer to each other are more likely to share similar structures). Such continuity benefits the OFTD model by learning a continuous and robust spatial-temporal function that enables more accurate completion results. The smooth bound is related to several factors (such as ω 0 and w). We experimentally evaluate such relationships in supplementary.
Based on Lemma 1, we present a regret bound of OFTD regarding the loss of historical information when fitting new data streams, which reveals the forgetting behavior of INR. Such forgetting exhibits a unique characteristic-it is position-dependent due to the spatial-temporal continuity of the OFTD model (see Fig. 2).
Theorem 1 (Regret bound for online FTD).
Denote the OFTD model learned at the historical time point t by X t = {f Θn ([I t n ]; t)} 3 n=1 ∈ R I t 1 ×I t 2 ×I tfoot_1 , where f Θn ([I t n ]; t) denotes the n-th factor function at time t. Assume that • The OFTD model learned at the new time point t + 1 using (6) is invariant at the boundary 3 (I t 1 , I t 2 , I t 3 ), i.e., f Θn (I t n ; t + 1) = f Θn (I t n ; t) (n = 1, 2, 3). • Each element of the weight matrix of the d-layer INRs {f Θn (•)} follows i.i.d. N (0, w 2 ) with sine activation function σ
(•) = sin(ω 0 •). The ℓ 1 -norm of derivative of each factor INR ∥f ′ Θn (x)∥ ℓ1 is bounded by κ > 0.
Then for any δ ∈ (0, 1) and any historical position (i
1 , i 2 , i 3 ) (i n ≤ I t n ), the following regret bound between the new OFTD model X t+1 = f Θn ([I t+1 n ]; t + 1) 3 n=1 ∈ R I t+1 1 ×I t+1 2 ×I t+1 3 (Take an example I t+1 n = I t n + 1) and the old OFTD model X t holds with probability at least 1 -δ: |(X t+1 ) (i1,i2,i3) -(X t ) (i1,i2,i3) | ≤ C 2 max n (I t n + 1 -i n ),(8)
where
C 2 = 6(η 3d ω 3d-3 0 +κη 2d ω 2d-2 0 )max(I t 1 I t 2 , I t 1 I t 3 , I t 2 I t 3 ) and η = 2wr 2 + w ln 3d δ .
Theorem 1 shows that the regret bound (i.e., the degree of forgetting) is proportional to the positional distance (I t n + 1 -i n ) between the considered point (i 1 , i 2 , i 3 ) and the new data stream position I t n + 1. This indicates that information that is more distant from the new data stream is more likely to be forgotten (see Fig. 2(a)). To alleviate the forgetting, we design a long-tail memory replay continual learning method.
this section cite: []

Section: Continual Learning via Memory Replay
OFTD transforms the classical online tensor decomposition into a continual learning paradigm, which expects to use the INRs to fit new data without forgetting historical data. To alleviate forgetting, we design a long-tail memory buffer that utilizes a part of historical data when fitting new data. Theorem 1 shows that more distant information is more likely to be forgotten. Thus we consider the long-tail Beta distribution (see Fig. 2(b)) to construct a memory buffer, which stores more data that is distant from the new data stream. For multi-aspect streams, the memory buffer
M t ∈ R I t 1 ×•••×I t N at the time t is constructed through a sampling process on the historical data X t-1 by (M t ) (i1,••• ,i N ) = (X t-1 ) (i1,••• ,i N ) if (i 1 , • • • , i N ) ∈ I t , otherwise (M t ) (i1,••• ,i N ) = 0, where I t = (i 1 , . . . , i N ) i n =⌊u j n I t-1 n ⌋, u j n ∼ Beta(α, β), j = 1, • • • , J, n = 1, • • • , N .
Here, ⌊•⌋ denotes round-down and the p.d.f. of Beta(α, β) is Beta(x; α, β) = Γ(α+β) Γ(α)Γ(β) x α-1 (1x) β-1 , x ∈ (0, 1). The I t is the index set of the memory buffer such that the indexes in I t follow a long-tail Beta distribution to store more information that is distant from the new data stream. This is achieved by setting appropriate α and β such that Beta(α, β) is a long-tail distribution (see Fig. 2(b)). A total number of J N indexes in I t are selected to construct the memory buffer M t . For single-aspect streams, we perform sampling on the streaming dimension and store all indexes for other dimensions. ) Involving the memory buffer during online optimization allows our model to retain historical data in M t over time, preventing forgetting. By reformulating the online model ( 6), the new online optimization model at time t with the memory buffer M t is formulated as
min {Θn} P t ⊗ Y t -f Θn ([I t n ]) N n=1 2 F + v∈It P t(v) ⊗ M t(v) -f Θn (v (n) ) N n=1 2 .
(9) Here, Y t contains new data at time t and M t includes a part of historical data to avoid forgetting. The more historical data in M t (i.e., the larger index set I t ), the more computational costs are needed for optimization. If we use all historical data to construct the memory buffer M t , then OFTD degrades to the batch FTD method that processes the whole tensor X t at each time. We summarize the OFTD algorithm with memory replay in Algorithm A.1 of the Appendix.
Given a tensor of size I 1 × • • • × I N , OFTD consumes O(mrd
N n=1 I n + r N n=1 I n ) in each iteration, where (mrd N n=1 I n ) is the INR complexity, (r N n=1 I n )
is the CP product complexity, r is the CP rank, and m, d are network's width and depth. We further propose a sharing strategy to improve the computational efficiency of OFTD, as detailed in Appendix A.2.
To further improve the effectiveness of OFTD for recovering highly irregular data streams (such as dynamic background and abrupt changes in data streams), we further proposed a temporal online affine regularizer to address this special situation, which is introduced in the Appendix A.3.
this section cite: []

Section: Experiments
We perform numerical experiments for both single-aspect and multi-aspect streaming data completion. We use the normalized reconstruction error (NRE) Ahn et al. [2021] for evaluation. The datasets are summarized in Appendix C.1. We consider the sampling rates ( SRs
this section cite: ['b3']

Section: Experimental Results
The quantitative results for single-aspect and multi-aspect streaming data completion are shown in Tables 2 and 3. OFTD attains better NRE results in most cases, showcasing its strong representation abilities and effectiveness for modeling tensor streams. This can be attributed to the compact, low-rank representation of CP decomposition and the expressive power of INRs to model dynamic structures of data streams. The OFTD achieves real-time updating with each online step costing less than 0.2/1.0 (single/multi-aspect) seconds. Also, from Fig. 3 it can be observed that OFTD better reconstructs the temporal curves of tensor streams, showcasing its capability to model complex data structures and preserve the temporal smoothness of the tensor. Overall, OFTD serves as a new state-of-the-art  online method for both single-aspect and multi-aspect streaming data completion. We show more experimental results in Appendix D.
this section cite: []

Section: Ablation Study
The ablation studies include the tests for the memory buffer size J N in Table 4 and the Beta distribution parameter β (with α fixed to 1) in Table 5. The memory buffer is effective to alleviate forgetting (compared to 0% in Table 4), thus enhancing performances. However, a large buffer size leads to increased computational costs, and we have set the buffer size to 33% of the whole tensor size in experiments. When the Beta distribution parameter β > 1, we obtain the desired long-tail memory buffer, resulting in good performance (Table 5) and justifying our memory buffer design. More ablation results include the CP-rank r, the usage of INR and its parameters are shown in Appendix D.
this section cite: []

Section: Conclusion
We have proposed a novel streaming data completion method OFTD, which utilizes CP decomposition parameterized by INRs to model tensor streams. Future work can be considered to design other advancing continual learning methods to enable life-long learning of INRs for the streaming data completion problem. For example, we can consider using regularization-based Sun et al. [2023] or gradient projection-based Lin et al. [2022] methods to further boost the performance of OFTD under the continual learning framework. Also, applying the low-rank functional parameterization using INRs paves a novel paradigm for low-rank adaptation (LoRA) of large models, which can be considered in future work.
this section cite: ['b41', 'b22']

Section: References
Ref_id:b0 Title: A contemporary and comprehensive survey on streaming tensor decomposition Year: (2022)
Ref_id:b1 Title: Robust tensor tracking with missing data and outliers: Novel adaptive CP decomposition and convergence analysis Year: (2022)
Ref_id:b2 Title: Tracking online low-rank approximations of higher-order incomplete streaming tensors Year: ()
Ref_id:b3 Title: Accurate online tensor factorization for temporal tensor streams with missing values Year: (2021)
Ref_id:b4 Title: Prototypesample relation distillation: towards replay-free continual learning Year: (2023)
Ref_id:b5 Title: Sal: Sign agnostic learning of shapes from raw data Year: (2020)
Ref_id:b6 Title: Online identification and tracking of subspaces from highly incomplete information Year: (2010)
Ref_id:b7 Title: Implicit neural spatial representations for time-dependent PDEs Year: (2023)
Ref_id:b8 Title: Learning continuous image representation with local implicit image function Year: (2021)
Ref_id:b9 Title: Petrels: Parallel subspace estimation and tracking by recursive least squares from partial observations Year: (2013)
Ref_id:b10 Title: Probabilistic streaming tensor decomposition Year: (2018)
Ref_id:b11 Title: Streaming bayesian deep tensor factorization Year: (2021)
Ref_id:b12 Title: Grassmannian optimization for online tensor completion and tracking with the t-svd Year: (2022)
Ref_id:b13 Title: Incremental gradient on the Grassmannian for online foreground and background separation in subsampled video Year: (2012)
Ref_id:b14 Title: Degradation accordant plug-and-play for low-rank tensor completion Year: (2022)
Ref_id:b15 Title: Static and streaming tucker decomposition for dense tensors Year: (2023)
Ref_id:b16 Title: Local implicit grid representations for 3D scenes Year: (2020)
Ref_id:b17 Title: Fast online low-rank tensor subspace tracking by CP decomposition using recursive least squares from incomplete observations Year: (2019)
Ref_id:b18 Title: Tensor decompositions and applications Year: (2009)
Ref_id:b19 Title: A novel recursive least-squares adaptive method for streaming tensor-train decomposition with incomplete observations Year: (2024)
Ref_id:b20 Title: Robust factorization of real-world tensor streams with patterns, missing values, and outliers Year: (2021)
Ref_id:b21 Title: Superpixel-informed implicit neural representation for multi-dimensional data Year: (2024)
Ref_id:b22 Title: TRGP: Trust region gradient projection for continual learning Year: (2022)
Ref_id:b23 Title: Robust online tensor completion for IoT streaming data recovery Year: (2022)
Ref_id:b24 Title: Finer: Flexible spectral-bias tuning in implicit neural representation by variableperiodic activation functions Year: (2024)
Ref_id:b25 Title: Subspace learning and imputation for streaming big data matrices and tensors Year: (2015)
Ref_id:b26 Title: Snapshot compressive imaging using domain-factorized deep video prior Year: (2024)
Ref_id:b27 Title: Local light field fusion: Practical view synthesis with prescriptive sampling guidelines Year: (2019)
Ref_id:b28 Title: NeRF: Representing scenes as neural radiance fields for view synthesis Year: (2021)
Ref_id:b29 Title: Adaptive parafac decomposition for third-order tensor completion Year: (2016)
Ref_id:b30 Title: Outlier-robust multi-aspect streaming tensor completion and factorization Year: (2019)
Ref_id:b31 Title: Inductive framework for multiaspect streaming tensor completion with side information Year: (2018)
Ref_id:b32 Title: Continual lifelong learning with neural networks: A review Year: (2019)
Ref_id:b33 Title: On and beyond total variation regularization in imaging: the role of space variance Year: (2023)
Ref_id:b34 Title: Multi-version tensor completion for time-delayed spatio-temporal data Year: ()
Ref_id:b35 Title: Wire: Wavelet implicit neural representations Year: (2023)
Ref_id:b36 Title: Improved implicit neural representation with fourier reparameterized training Year: (2024)
Ref_id:b37 Title: Implicit neural representations with periodic activation functions Year: (2020)
Ref_id:b38 Title: Adversarial generation of continuous images Year: (2021)
Ref_id:b39 Title: Streaming tensor factorization for infinite data sources Year: (2018)
Ref_id:b40 Title: Multi-aspect streaming tensor completion Year: (2017)
Ref_id:b41 Title: Regularizing second-order influences for continual learning Year: (2023-06)
Ref_id:b42 Title: Online tensor low-rank representation for streaming data clustering Year: (2022)
Ref_id:b43 Title: eOTD: An efficient online tucker decomposition for higher order tensors Year: (2018)
Ref_id:b44 Title: Gocpt: Generalized online canonical polyadic tensor factorization and completion Year: (2022)
Ref_id:b45 Title: DisMASTD: An efficient distributed multiaspect streaming tensor decomposition Year: (2023)
Ref_id:b46 Title: Effective streaming low-tubal-rank tensor approximation via frequent directions Year: (2022)
Ref_id:b47 Title: Accelerated online low rank tensor learning for multivariate spatiotemporal streams Year: (2015)
Ref_id:b48 Title: An online tensor robust PCA algorithm for sequential 2d data Year: (2016)
Ref_id:b49 Title: Quantum implicit neural representations Year: (2024)
Ref_id:b50 Title: Window-based dynamic streaming tensor analysis based on CP decomposition Year: (2021)
