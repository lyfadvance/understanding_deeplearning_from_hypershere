# Understanding Deep Learning from the Perspective of Hyperspherical Non-Euclidean Geometry

**Abstract**: This paper aims to establish a profound connection between Euclidean space and hyperspherical geometry within deep learning models. We demonstrate that by mapping data to high-dimensional hyperspheres and utilizing stereographic projection to equivalently transform spherical decision boundaries in Euclidean space into linear (hyperplane) decision boundaries on hyperspheres, a theoretically guaranteed representation learning framework can be constructed. Within this framework, we prove the existence of a class of ball-preserving nonlinear transformations. These transformations can perform lossless rearrangement of data and ultimately render any labeled binary classification dataset linearly separable. Furthermore, we explore the conditions for maintaining hyperspherical structure under multi-hyperplane partitioning and connect this to the theory of invertible neural networks. This work provides a unified geometric perspective for understanding the representational capacity, optimization landscape, and design of novel architectures in deep neural networks.

**Keywords**: Hyperspherical Geometry; Universal Approximation; Representation Learning; Stereographic Projection; Invertible Neural Networks; Deep Learning Theory

## Introduction

The success of deep learning is often attributed to the powerful representational capacity of deep neural networks. However, their internal working mechanisms, particularly the evolving geometry of data representations during training, remain unclear. Traditional analyses are mostly based on Euclidean space. This paper shifts the perspective to the non-Euclidean geometry of hyperspheres, aiming to reveal its intrinsic connection to the core principles of deep learning.

The core hypothesis is that an ideal representation learning process should be able to progressively reshape the feature space—without destroying the overall structure of the data (i.e., performing a "rearrangement")—ultimately achieving linear separability. We first construct a theoretical model in Euclidean space based on spherical partitioning to prove the existence of such a process. However, the precision required for spherical partitioning makes it difficult to learn. To address this, we introduce **stereographic projection**, which equivalently transforms spherical partitioning in Euclidean space into hyperplane partitioning on a hypersphere. This transformation is crucial; it shifts the problem from finding the precise center and radius of a sphere to learning a direction (normal vector) on the hypersphere, which is "easier" in high-dimensional spaces (see Chapter 3 on Easy Learning).

Based on this, we prove the existence of a ball-preserving nonlinear transformation. This transformation is composed of a series of data-dependent conditional orthogonal transformations (e.g., Householder reflections). It can adjust the relative positions of data points while keeping them on the hypersphere, progressively achieving the goal of linear separability. Finally, we generalize this framework to the case of multi-hyperplane partitioning and find that its structure coincides with the design of coupling layers in invertible neural networks (INNs), thereby offering new insights into the geometric interpretation of such models.

### **Related Work**

In the theoretical landscape of geometric deep learning, the hyperspherical manifold has become a crucial cornerstone for understanding and designing neural network representations, owing to its unique symmetry and compactness. Existing research primarily unfolds from two interrelated yet distinct perspectives: first, treating the hypersphere as a **natural space for data representation** to design loss functions and optimization objectives for improving model discriminative performance; second, treating it as a **carrier for symmetry actions** to construct network architectures with specific equivariance properties for embedding the inductive biases of the physical world. This section will review these works and position the theoretical starting point of our research.

#### **1. Hyperspherical Representation and Discriminative Learning**

A series of influential works have shown that constraining or mapping features onto a hypersphere can significantly enhance the discriminative ability of deep learning models, particularly in metric learning tasks. The core of this research lies in using **angular distance** or **geodesic distance** on the hypersphere to replace traditional Euclidean distance, thereby obtaining a more discriminative and semantically consistent feature space. For instance, in face recognition, a series of loss functions based on angular margin (e.g., SphereFace, CosFace, ArcFace) have been proposed. By directly applying constraints on the hyperspherical manifold, they achieve more compact intra-class features and greater separation between inter-class features, leading to breakthrough results.

Recent research has further explored the impact of the **mapping process from data space to hypersphere** itself on learning. The paper "*Improvising the Learning of Neural Networks on Hyperspherical Manifold*" is a typical example in this direction. This work systematically applies **stereographic projection** as a pre-processing transformation module to standard angular margin loss functions. Its experiments show that explicitly mapping data from Euclidean space to the hyperspherical manifold can serve as an effective inductive bias, improving model performance on image classification datasets like CIFAR. This work reveals that **manifold transformation** has a positive guiding effect on the optimization process. However, its main contribution lies in the empirical performance improvement, without delving into the fundamental theoretical issues regarding the ultimate limits of representational learning capability implied by such transformations.

#### **2. Symmetry and Equivariant Architectures on Hyperspheres**

Another important research thread stems from the basic idea of geometric deep learning: encoding the intrinsic symmetries of data (e.g., translation, rotation, reflection) into the network structure itself to build **equivariant** or **invariant** models. This idea is systematically expounded in survey works like "*Geometric deep learning: Grids, groups, graphs, geodesics, and gauges*". Within this framework, the hypersphere, due to its rich symmetry group (the n-dimensional orthogonal group O(n)), becomes an ideal object for constructing equivariant networks.

The representative work "*On Learning Deep O(n)-Equivariant Hyperspheres*" focuses on constructing deep learning layers that are strictly equivariant to O(n) group transformations (i.e., n-dimensional rotations and reflections). Utilizing geometric tools like spherical neurons and regular simplices, this work designs network components to ensure their outputs change predictably and coherently with orthogonal transformations of the input. Such methods show significant advantages in domains like point cloud analysis and molecular modeling, where **symmetry priors are crucial**. Their core contribution lies in providing a structural, scalable solution for incorporating inductive bias.

Furthermore, **invertible neural networks** (e.g., RevNet), aimed at preserving information flow, offer insights from another angle. By designing bijective network layers, such models theoretically guarantee lossless information flow during forward propagation. This shares a philosophical connection with the idea of transformations that preserve metrics or volume on a manifold.

#### **3. Local Translation Transformations in Euclidean Space**

From the perspective of this paper, the residual method of ResNet can be seen as performing local translation transformations in Euclidean space. This paper extends this to local translation, reflection, and rotation transformations on the hyperspherical manifold via Householder reflections.

#### **4. Positioning of This Research**

The aforementioned works have advanced hyperspherical deep learning from the two dimensions of **engineering optimization for discriminative performance** and **structural embedding of symmetry**. However, a more fundamental theoretical question remains: apart from specific task objectives or predefined symmetries, what is the ultimate **limit of morphological shaping capability** for the representation space itself through a series of nonlinear transformations?

This is precisely the perspective from which our research originates. Unlike the paths of optimizing specific loss functions or imposing symmetry constraints, this paper aims to construct a framework for investigating the **theoretical reachability** of representation learning. We prove that after mapping data to a hypersphere via stereographic projection, there exists a series of **ball-preserving (i.e., maintaining manifold structure) conditional orthogonal transformations** guided by hyperplane partitioning. These transformations can achieve lossless rearrangement of data representations and inevitably lead to a linearly separable representation space. The core of this framework lies in **"transformation"** and **"reachability"**. It does not aim to propose an alternative practical loss function or network layer but provides an abstract yet theoretically complete lens for understanding how the complex layered operations in deep networks collaboratively work to shape the representation space. This foundational inquiry also offers deeper theoretical insights into the performance improvement phenomena observed in the aforementioned application-oriented research.

**Kernel Methods and Feature Space**: Our stereographic projection $\Psi$ is similar to the feature mapping in kernel methods, which implicitly maps data to high-dimensional (even infinite-dimensional) spaces to linearize problems [6]. This paper clarifies that the geometric goal of such mapping is to obtain a hyperspherical representation.

**Metric Perspective on Representation Learning**: The view of the learning process as "rearrangement" or "metric reshaping" in the representation space aligns with some representation learning theories [8]. The hyperspherical perspective in this paper provides a geometrically tractable stage for theoretically analyzing this reshaping process.

## Universal Approximation Theorem Based on Rearrangement

### Spherical Partitioning in Euclidean Space

Consider a finite dataset $X = \{ \mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n \}$, where $\mathbf{x}_i \in \mathbb{R}^D$, and its corresponding binary labels $B = (b_1, b_2, ..., b_n)$, $b_i \in \{0, 1\}$. Our goal is to construct a model $f(\cdot; W)$ with parameters $W$, satisfying the following properties:

1.  **Permutation Preservation**: For any parameter $W$, the output $X' = f(X; W)$ is a permutation (rearrangement) of the input $X$.
2.  **Permutation Completeness**: For any permutation $X'_{\text{perm}}$ of $X$, there exists a parameter $W$ such that $f(X; W) = X'_{\text{perm}}$.
3.  **Linear Separability**: There exists a parameter $W^*$ such that the rearranged dataset $X'^* = f(X; W^*)$ is linearly separable with respect to the labels $B$.

We first construct this within a framework that allows the use of the spherical sign function $\operatorname{sphereSign}(\mathbf{x}; \mathbf{a}, b) = \operatorname{sign}(\|\mathbf{x} - \mathbf{a}\|^2 - b)$, linear functions, conditional `IF` statements, and nearest neighbor `MIN` selection operations.

**Proof Outline**:

*   **Constructing Fixed Weight Space**: For each point $\mathbf{x}_k$ in the dataset, we can construct a spherical partition that uniquely identifies that point. A simple construction is to set the center $\mathbf{a}^k = \mathbf{x}^k$ and the radius $b = \epsilon$ (a very small positive number). All such $(\mathbf{a}^k, b)$ constitute the fixed weight space $S_{\text{fix}} = X$.
*   **Implementing Arbitrary Pairwise Exchange**: Each step of the model $f$ aims to exchange a pair of points $(\mathbf{x}^p, \mathbf{x}^q)$ in the dataset. The learnable parameters $w_p, w_q \in S_{\text{learn}}$ select the corresponding $(\mathbf{a}^p, b^p)$ and $(\mathbf{a}^q, b^q)$ from $S_{\text{fix}}$ via the `MIN` operation. Then, using `IF` statements and `sphereSign` checks, the point inside the sphere of $\mathbf{a}^q$ (i.e., $\mathbf{x}^q$) is moved to $\mathbf{a}^p$, and vice versa, completing the exchange. By composing multiple such exchange steps, any permutation can be realized.
*   **Proving Existence of a Linearly Separable Permutation**: For any $n$ distinct points in $\mathbb{R}^D$ and any integer $p$ ($0 \le p \le n$), there always exists a hyperplane that exactly partitions the point set into two parts containing $p$ and $n-p$ points, respectively [1]. Since our labels $B$ partition the data into two sets (of sizes $p$ and $n-p$), a hyperplane separating these two sets always exists. Consequently, using the exchange operations described above, we can always learn a permutation $W^*$ that moves points with the same label to the same side of the hyperplane, thereby achieving linear separability.

The detailed proof process is given below.

Assume the weights consist of fixed weight parameters $W_{ul} = X$ and learnable weight parameters $W_{l}$. This approach, combined with the $MIN_x$ operation, allows $W_l$ to learn within the entire space.

It is evident that for any $x^k \in X$ in the dataset, there exist $a^k, b^k$ such that:
$$
\begin{align}
\operatorname{sphereSign}(x_m, a^k, b^k) > 0 \quad & \text{if } k = m \\
\operatorname{sphereSign}(x_m, a^k, b^k) < 0 \quad & \text{if } k \neq m
\end{align}
$$
For all $x^k$, we construct such $a^k, b^k$. All these $a^k , b^k$ constitute the fixed weight space $SPACE_{fix} = \{(a^k, b^k) | k = 1...n\}$. We can directly assume $a^k = x^k, b= \epsilon$, so this fixed weight space is essentially the dataset X itself.

Next, we construct the model structure $f$ such that each step exchanges two data points in the dataset. Therefore, the learnable weights for each step are $w_p$ and $w_q$, with $w_p, w_q \in SPACE_W$. All learnable weights for each step constitute $SPACE_{learn}$.

Suppose the data to be exchanged in the current step are $x^p$ and $x^q$. The procedure is as follows:

> $a^q, b^q = MIN(w_q, SPACE_{fix})$
>
> $a^p, b^p = MIN(w_p, SPACE_{fix})$
>
> `IF` $\operatorname{sphereSign}(x, a^q, b^q)$:
>
> ​	$x = a^p$
>
> `ELSE IF` $\operatorname{sphereSign}(x, a^p, b^p)$:
>
> $x = a^q$
>
> `ELSE`:
>
> $x = x$

Through this operation, the dataset is rearranged. The above process ensures that:

1.  For any $W$, the output $X' = f(X,W)$ is necessarily a permutation of $X$.
2.  For any $X'$ that is a permutation of $X$, there exists $W$ such that $f(X,W)=X'$.

Now, we prove the following theorem:

**Theorem**: Let $ S=\{x_1,x_2,…,x_m\} $ be $m$ distinct points in $R_n$. Then, for any integer $p(0≤p≤m)$, there always exists a hyperplane $H$ dividing the space into two open half-spaces such that one half-space contains exactly $p$ points, the other contains $m−p$ points, and no point lies on the hyperplane $H$.

**Step 1: Existence of a "Good" Direction**
We need to find a unit vector $u∈R_n$ such that for any two distinct points $x_i≠x_j$, we have $u⋅x_i≠u⋅x_j$.

*   **Reason**: For any fixed pair $(i,j)$, the set of $u$ satisfying $u⋅(x_i − x_j)=0$ is a hyperplane through the origin in n-dimensional space (i.e., the set of all vectors perpendicular to $(x_i−x_j)$). Its intersection with the unit sphere is a "great circle" (or a lower-dimensional sphere), which has **measure zero**.
*   Since there are only finitely many pairs $(i,j)$ (a total of $C_m^2$ pairs), the union of all "bad directions" (directions that cause any two points to have the same projection) is a finite union of zero-measure sets, which is **still a set of measure zero**.
*   Therefore, on the unit sphere, **almost all** directions are "good directions". We can arbitrarily choose one (e.g., generate a random vector; it will almost surely satisfy the condition).

**Step 2: Projection and Sorting**
Fix such a "good direction" $u$. Compute the projection (scalar) of each point onto this direction:
$$
y_i=u⋅x_i,\quad i=1,2,…,m.
$$
By Step 1, all $y_i$ are pairwise distinct. Sort them in strictly increasing order:
$$
y_{(1)}<y_{(2)}<⋯<y_{(m)}.
$$
Here, $y_{(k)}$ corresponds to some point $x_{(k)}$. This order is a **total order** of the point set along the direction $u$.

**Step 3: Constructing the Hyperplane**
Now, for a given integer $p(0≤p≤m)$:

1.  When $1≤p≤m−1$, choose a real number $t$ such that $y_{(p)}<t<y_{(p+1)}$.
2.  When $p=0$, choose $t<y_{(1)}$.
3.  When $p=m$, choose $t>y_{(m)}$.

Consider the hyperplane $H$ defined by the equation $u⋅x=t$.

**Step 4: Verification of Partition**

*   **No point lies on the plane**: For any point $x_i$, its projection $y_i$ is either less than $t$ or greater than $t$, because $t$ is not equal to any $y_i$. Therefore, $u⋅x_i≠t$, meaning no point lies on $H$.
*   **Exact count**:
    *   For points satisfying $u⋅x_i<t$, their projections $y_i$ are less than $t$. According to the choice of $t$, **there are exactly $p$ such points** (i.e., points with projections $y_{(1)},y_{(2)},…,y_{(p)}$).
    *   The remaining $m−p$ points must satisfy $u⋅x_i>t$, lying on the other side of the hyperplane.

Thus, the hyperplane $H$ strictly partitions the point set into two parts containing $p$ and $m−p$ points respectively, completing the proof.

至此, we have proven that in Euclidean space, a model based on spherical partitioning can satisfy the three aforementioned properties. However, this construction relies on precise memorization and matching of data points, and the parameters of the spherical partition (center and radius) are difficult to learn effectively via gradient descent from random initialization in high-dimensional spaces.

### Transformation to Hyperspherical Geometry: Stereographic Projection

To address the "hard to learn" issue of spherical partitioning, we introduce stereographic projection to transform the problem into hyperspherical geometry. The core insight is: **Spherical decision boundaries in $n$-dimensional Euclidean space can be equivalently transformed into hyperplane decision boundaries on an $n$-dimensional hypersphere.**

**Construction**: Define the mapping $\Phi: \mathbb{R}^n \rightarrow \mathbb{R}^{n+2}$,
$$
\Phi(\mathbf{x}) = (1, \mathbf{x}, \|\mathbf{x}\|^2).
$$
Normalizing yields the stereographic projection $\Psi: \mathbb{R}^n \rightarrow S^{n+1} \subset \mathbb{R}^{n+2}$,
$$
\Psi(\mathbf{x}) = \frac{\Phi(\mathbf{x})}{\|\Phi(\mathbf{x})\|}.
$$

**Equivalence**: Consider the spherical partition $\operatorname{sign}(\|\mathbf{x} - \mathbf{a}\|^2 - b)$ in Euclidean space. Expanding it:
$$
\|\mathbf{x} - \mathbf{a}\|^2 - b = \|\mathbf{x}\|^2 - 2\mathbf{a}\cdot\mathbf{x} + \|\mathbf{a}\|^2 - b.
$$
Define the vector $\mathbf{w} = (\|\mathbf{a}\|^2 - b, -2a_1, ..., -2a_n, 1) \in \mathbb{R}^{n+2}$. Then we have:
$$
\mathbf{w} \cdot \Phi(\mathbf{x}) = \|\mathbf{x} - \mathbf{a}\|^2 - b.
$$
Since $\Phi(\mathbf{x}) = \|\Phi(\mathbf{x})\| \Psi(\mathbf{x})$ and $\|\Phi(\mathbf{x})\| > 0$, we obtain:
$$
\operatorname{sign}(\|\mathbf{x} - \mathbf{a}\|^2 - b) = \operatorname{sign}(\mathbf{w} \cdot \Psi(\mathbf{x})).
$$
Therefore, the spherical partition $\operatorname{sign}(\|\mathbf{x} - \mathbf{a}\|^2 - b)$ in Euclidean space is equivalent to the hyperplane partition $\operatorname{sign}(\mathbf{w} \cdot \Psi(\mathbf{x}))$ on the hypersphere $S^{n+1}$.

This transformation is significant. It means that under the hyperspherical representation, we only need to use a **simple linear discriminant function** $\operatorname{sign}(\mathbf{w} \cdot \mathbf{z})$ (where $\mathbf{z} \in S^{n+1}$) to achieve the expressive power of the complex spherical partitioning in the original Euclidean space. Combining with the argument in Section 2.1, we obtain the following theorem:

**Theorem 2.1 (Hyperspherical Rearrangement and Linear Separability)**: There exists a model $f$ that first maps the data $X \subset \mathbb{R}^D$ to the hypersphere $S^{D+1}$ via stereographic projection $\Psi$, and then uses only linear discriminant functions, conditional operations, and nearest neighbor selection, such that:
      1. For any parameter $W$, the output is a permutation of the input data on the hypersphere.
      2. For any permutation, there exists a parameter $W$ that realizes it.
      3. There exists a parameter $W^*$ such that the rearranged data representation is linearly separable.

## The Geometric Perspective of Easy Learning

Section 2.1 pointed out that directly learning precise spherical partitions in Euclidean space is difficult. We formalize the concept of "Easy Learning": Assume model parameters $W$ are initialized with $N(0, \sigma^2 I)$, whose distribution approximates a hypersphere $S_{\text{param}}^m$ in high-dimensional space. If for almost any point on $S_{\text{param}}^m$ (i.e., randomly initialized $W$), there exists a small perturbation $\Delta W$ such that $W + \Delta W$ is a local minimum for some problem solution, then that problem is easy to learn.

Spherical partitioning is difficult because its solutions (specific $(\mathbf{a}, b)$) are like "isolated islands" in the high-dimensional parameter space; the probability of randomly initialized $W$ landing near a solution is extremely low. In contrast, for hyperplane partitioning $\operatorname{sign}(\mathbf{w} \cdot \mathbf{z})$, the "direction" of the solution (normal vector $\mathbf{w}$) is crucial, while its magnitude does not affect the sign function. On a high-dimensional hypersphere, almost all directions can be potential "good" initial directions (see the discussion on "good directions" in the proof of hyperplane existence in Section 2.1), and it is easier to reach the target direction via rotation (gradient updates). Therefore, using hyperplane partitioning under the hyperspherical representation aligns better with the intuition of "Easy Learning".

## Ball-Preserving Nonlinear Transformations on Hyperspheres

The construction in Theorem 2.1 requires memorizing the entire dataset and lacks generalization ability. We shift to a more realistic setting: assume the data is uniformly distributed (or originates from) a hypersphere $S^{m-1}$. We investigate whether there exists a parameterized nonlinear transformation $T: S^{m-1} \rightarrow S^{m-1}$ that not only keeps the data on the sphere (ball-preserving) but also, by adjusting parameters, renders the transformed data linearly separable for any given binary classification labels.

### Transformation Under Single-Hyperplane Partitioning

A natural construction is to iteratively perform the following two-step operation:

1.  **Partitioning**: Partition the hypersphere $S^{m-1}$ into two hemispheres using a hyperplane $H: \mathbf{a} \cdot \mathbf{z} = 0$ (passing through the sphere's center).
2.  **Conditional Rotation**: Select one of the hemispheres (a symmetric geometric body) and apply a rotation around its symmetry center (i.e., the sphere's center). This can be achieved via Householder reflection. Householder reflections hold an exceptionally important place in hyperspherical geometry. Sequential Householder reflections can accomplish all translation, rotation, and reflection operations for a certain region in hyperspherical geometry. Therefore, we use Householder reflections to implement the operation of rotating a symmetric geometric body around its symmetry center. Given two unit vectors $\mathbf{u}', \mathbf{v}'$ orthogonal to $\mathbf{a}$, the generated rotation can be represented as the composition of two Householder reflections $H_{\mathbf{u}'} H_{\mathbf{v}'}$. This operation acts only on the selected hemisphere.

The transformation $T$ is clearly ball-preserving. Intuitively, through a series of such "cut-and-rotate" operations, we can gradually separate data points belonging to different classes into different regions on the sphere—like stirring milk foam in coffee—until a hyperplane exists that can separate them. A rigorous proof requires more complex geometric arguments. Since proving the continuous case is complex, we assume the number of elements in the dataset is finite and provide a proof for the discrete case below, as follows:

#### Theorem 1 (Minimal Step-Size Single Point Movement)

**First Step**: Prove that for a dataset whose points lie on a hypersphere, for any data point $p$ on the hypersphere, there exists a geodesic circle centered at that point containing only that point. Furthermore, for any point in this geodesic circle taken as the normal vector, the hyperspherical cap (spherical cap) cut by the hyperplane passing through $p$ and having that vector as normal also contains only that point.

Let dataset $X$ be on the hypersphere $S^{d-1} \subset \mathbb{R}^d$, and assume $X$ is discrete (the geodesic distance between any two points has a positive lower bound). For any point $p \in X$, define:
$$
\delta = \min\{ d(p, x) : x \in X, x \neq p \} > 0.
$$
Choose $r$ satisfying $0 < r < \delta/2$. Consider the geodesic disk centered at $p$ with radius $r$:
$$
B(p, r) = \{ x \in S^{d-1} : d(p, x) < r \}.
$$
Since $r < \delta$, we have $B(p, r) \cap X = \{p\}$.

Now, consider the geodesic circle:
$$
C(p, r) = \{ q \in S^{d-1} : d(p, q) = r \}.
$$
For any $q \in C(p, r)$, construct the hyperplane:
$$
H_q = \{ x \in \mathbb{R}^d : (x - p) \cdot q = 0 \}.
$$
It has $q$ as its normal vector and passes through point $p$. This hyperplane divides the hypersphere into two closed spherical caps:
$$
C_q^+ = \{ x \in S^{d-1} : (x - p) \cdot q \geq 0 \}, \quad C_q^- = \{ x \in S^{d-1} : (x - p) \cdot q \leq 0 \}.
$$
We now prove $C_q^+ \cap X = \{p\}$.

Let $x \in X \setminus \{p\}$, and let $\theta = d(p, x) \geq \delta > 2r$. Treating points as unit vectors, we have $p \cdot q = \cos r$. By the spherical triangle inequality, $d(q, x) \in [\theta - r, \theta + r]$, thus:
$$
\cos(\theta + r) \leq x \cdot q \leq \cos(\theta - r).
$$
Since $\theta > 2r$, we have $\theta - r > r$, and the cosine function is monotonically decreasing on $[0, \pi]$, so:
$$
x \cdot q \leq \cos(\theta - r) < \cos r = p \cdot q.
$$
Therefore, $(x - p) \cdot q < 0$, i.e., $x \notin C_q^+$. Hence, $C_q^+ \cap X = \{p\}$.

In summary, for any point $p \in X$, there exists a radius $r > 0$ such that the geodesic disk $B(p, r)$ centered at $p$ contains only $p$, and for any point $q$ on this geodesic circle $C(p, r)$, the spherical cap $C_q^+$ cut by the hyperplane with normal $q$ passing through $p$ also contains only $p$. Q.E.D.

**Second Step**: Let the constant $\varepsilon_0$ be the radius of the geodesic circle taken in the first step, i.e., $\varepsilon_0 = r$. Then for any point $q \in S^{d-1}$ satisfying $0 <  \text{dist}(p,q) \leq \varepsilon_0$, there exists a point $r \in C(p,\varepsilon_0)$ and a rotation $R$ (with rotation axis in the geodesic circle and rotation angle $\text{dist}(p,q)$) such that $R(p) =  q$.

Construct the geodesic from $p$ to $q$. Use the midpoint of the geodesic to construct $w$, and half the length of the geodesic to construct $b$. Then, using $W$ as the rotation center and selecting an additional rotation plane, perform a 180-degree rotation to rotate $P$ to $Q$.

#### Theorem 2 (Finite-Step Linear Separability)

For any finite labeling configuration $f:X\to\{0,1\}$, there exists a finite number of the above $\varepsilon_0$-moves such that ultimately all points labeled $1$ lie within a common hemisphere (i.e., are linearly separable).

**Proof**:

*   With $\varepsilon_0$ as the step size, the shortest geodesic length from any $p\in X$ to any target $q\in\mathbb{S}^{d-1}$ is $\leq\pi$.
*   The upper bound on the required number of steps is:
    $$
    N_{\max}=\left\lceil\frac{\pi}{\varepsilon_0}\right\rceil
    $$
    (depends on $X$, but not on the specific $p,q$).
*   Perform chained $\varepsilon_0$-step moves for all "Southern hemisphere" points labeled $1$, pushing each near the North Pole one by one.
*   Each point requires at most $N_{\max}$ steps.
*   The total number of steps is $\leq|X|\cdot N_{\max}$, which is finite.
*   Finally, all points labeled $1$ lie within a common hemisphere, achieving linear separability.

### Multi-Hyperplane Partitioning and Invertible Structure

Using only one hyperplane per step is inefficient. We hope each step can use multiple hyperplanes for finer partitioning. However, directly applying rotations to regions defined by multiple hyperplanes generally cannot guarantee that the overall transformation is ball-preserving.

**Key Question**: Under what conditions can conditional transformations under multi-hyperplane partitioning preserve the hypersphere?

**The answer stems from invertible neural networks** [2, 3]. Consider decomposing the space as $\mathbb{R}^m = \mathbb{R}^{m-k} \times \mathbb{R}^k$, denoting points as $\mathbf{z} = (\mathbf{x}, \mathbf{y})$. We restrict the normal vectors of the partitioning hyperplanes to depend only on the first $m-k$ coordinates, i.e., they have the form $\mathbf{a}^{(j)} = (\mathbf{a}_0^{(j)}, \mathbf{0})$. Then, the partitioning depends only on $\mathbf{x}$, producing regions $\{A_\ell\}$. On the hypersphere $S^{m-1}$, the corresponding regions are:
$$
\tilde{R}_\ell = \{ (\mathbf{x}, \mathbf{y}) \in S^{m-1} | \mathbf{x} \in A_\ell \}.
$$
For a fixed $\mathbf{x}$, the $\mathbf{y}$ part of the point $(\mathbf{x}, \mathbf{y})$ lies on a $(k-1)$-dimensional sphere $S^{k-1}(\sqrt{1-\|\mathbf{x}\|^2})$. This structure is called a **local product structure**.

Within this structure, we can define a ball-preserving transformation:
$$
T(\mathbf{x}, \mathbf{y}) = (\mathbf{x}, O_\ell(\mathbf{x}) \mathbf{y}), \quad \text{if } (\mathbf{x}, \mathbf{y}) \in \tilde{R}_\ell.
$$
where $O_\ell(\mathbf{x}) \in O(k)$ is an orthogonal matrix dependent on $\mathbf{x}$ (and thus on region $\ell$). Since orthogonal transformations preserve the norm of $\mathbf{y}$, we have $\|T(\mathbf{x}, \mathbf{y})\| = \|\mathbf{z}\| = 1$, so transformation $T$ is ball-preserving. In particular, if $O_\ell(\mathbf{x})$ is generated by a Householder reflection $H_{\mathbf{u}(\mathbf{x})}$, and the reflection vector $\mathbf{u}$ has the form $\mathbf{u} = (0, \mathbf{u}_1)$, then the transformation can be written as:
$$
\begin{aligned}
\mathbf{x}' &= \mathbf{x}, \\
\mathbf{y}' &= H_{f(\mathbf{x})} \mathbf{y}, \quad \text{where } f(\mathbf{x}) = \operatorname{sign}(\mathbf{a}_0 \cdot \mathbf{x} + b) \mathbf{u}_1.
\end{aligned}
$$
This is precisely the core idea of **coupling layers in invertible neural networks** (e.g., RealNVP [4], GLOW [5]): one part of the dimensions ($\mathbf{x}$) remains unchanged and is used to generate parameters (here, the reflection vector) that control the transformation of the other part ($\mathbf{y}$). Our analysis provides a clear geometric interpretation: **Invertible coupling layers essentially apply ball-preserving, conditional orthogonal transformations on regions with a local product structure.**

The following provides a detailed explanation of this process.

Consider the space $\mathbb{R}^{m} = \mathbb{R}^{m-k} \times \mathbb{R}^{k}$, denote points as $z = (x,y)$, where $x \in \mathbb{R}^{m-k}, y \in \mathbb{R}^{k}$. The unit hypersphere is:
$$
S^{m-1} = \{(x,y) \in \mathbb{R}^{m} : \|x\|^{2} + \|y\|^{2} = 1\}.
$$
Assume a set of hyperplanes, each defined by a normal vector $a^{(j)} \in \mathbb{R}^{m}$ and a bias $b_{j} \in \mathbb{R}$, with equation:
$$
a^{(j)} \cdot z = b_{j}, \quad j=1,\ldots,N.
$$
Key restriction: All normal vectors satisfy $a^{(j)} = (a^{(j)}_{0},0)$, where $a^{(j)}_{0} \in \mathbb{R}^{m-k}$. That is, the hyperplane equations involve only the first $m-k$ coordinates:
$$
a^{(j)}_{0} \cdot x = b_{j}.
$$

These hyperplanes partition $\mathbb{R}^{m-k}$ into several convex polyhedral regions $\{A_{\ell}\}_{\ell \in L}$, each $A_{\ell}$ being an open set in $x$-space defined by a set of strict linear inequalities. In $\mathbb{R}^{m}$, the corresponding regions are:
$$
R_{\ell} = \{(x,y) \in \mathbb{R}^{m} : x \in A_{\ell}\}.
$$
On the hypersphere, the regions are:
$$
\tilde{R}_{\ell} = R_{\ell} \cap S^{m-1} = \{(x,y) \in S^{m-1} : x \in A_{\ell}\}.
$$
Fix $x \in A_{\ell}$ satisfying $\|x\| < 1$ (if $\|x\| = 1$, then $y = 0$, an isolated point). $y$ must satisfy $\|y\|^{2} = 1 - \|x\|^{2}$, i.e., $y$ belongs to the sphere of radius $\sqrt{1 - \|x\|^{2}}$, $S^{k-1}(\sqrt{1 - \|x\|^{2}})$. Therefore:
$$
\tilde{R}_{\ell} = \bigcup_{x \in A_{\ell} \cap B_{m-k}} \{x\} \times S^{k-1}\left(\sqrt{1 - \|x\|^{2}}\right),
$$
where $B_{m-k} = \{x \in \mathbb{R}^{m-k} : \|x\| \leq 1\}$. Each $\tilde{R}_{\ell}$ has a **local product structure**: the $x$ part belongs to $A_{\ell}$, and the $y$ part is a sphere (with dimension varying with $x$).

For each region $\tilde{R}_\ell$, define the transformation $T_\ell: \tilde{R}_\ell \rightarrow S^{m-1}$ as:
$$
T_\ell(x, y) = (x, O_\ell(x)y),
$$
where $O_\ell(x) \in O(k)$ is an orthogonal matrix dependent on $x$ (e.g., generated by a Householder reflection). Since $O_\ell(x)$ is orthogonal:
$$
\|O_\ell(x)y\| = \|y\|,
$$
and thus:
$$
\|x\|^2 + \|O_\ell(x)y\|^2 = \|x\|^2 + \|y\|^2 = 1,
$$
meaning $T_\ell(x, y) \in S^{m-1}$. The overall transformation $T: S^{m-1} \rightarrow S^{m-1}$ is the concatenation of all $T_\ell$:
$$
T(x, y) = T_\ell(x, y) \quad \text{when} \quad (x, y) \in \tilde{R}_\ell.
$$

*   **Well-definedness**: Since the regions $\{\tilde{R}_\ell\}$ partition $S^{m-1}$ (boundaries have measure zero and can be ignored), $T$ is defined almost everywhere.
*   **Range on the sphere**: As shown, each $T_\ell$ maps $\tilde{R}_\ell$ to a subset of $S^{m-1}$, so the image of $T$ is contained in $S^{m-1}$.
*   **Surjectivity**: For any $(x', y') \in S^{m-1}$, there exists $\ell$ such that $x' \in A_\ell$. Let $y = O_\ell(x')^{-1}y'$. Then $(x', y) \in \tilde{R}_\ell$ and $T(x', y) = (x', y')$, so $T$ is surjective.
*   **Injectivity**: If $T(x_1, y_1) = T(x_2, y_2)$, then $x_1 = x_2$ (since the $x$ part is unchanged), and $O_\ell(x_1)y_1 = O_\ell(x_1)y_2$. Since orthogonal matrices are invertible, $y_1 = y_2$.

Therefore, $T$ is a bijection from $S^{m-1}$ to itself and preserves the hypersphere.

**Invertibility and Geometric Interpretation**
Since $T$ is invertible at every point and its inverse is easy to compute ($T^{-1}(x, y) = (x, O_\ell(x)^{-1}y)$), this forms the basis of invertible neural networks. Geometrically, the hyperplane set $\{(a_i, 0)\}$ partitions the hypersphere into regions, each possessing $S^k$ symmetry for a fixed $x$: $y$ can rotate freely without changing region membership. Therefore, applying any orthogonal transformation to $y$ (dependent on $x$) does not destroy the hypersphere, and the region as a whole remains unchanged.

## 6. Conclusion and Future Work

This paper systematically expounds the important role of hyperspherical geometry in understanding deep learning. We have proven that via stereographic projection, complex partitioning problems in Euclidean space can be transformed into linear problems on hyperspheres. This transformation is not only theoretically complete but also aligns better with the characteristics of gradient descent optimization (Easy Learning). Furthermore, we have constructed ball-preserving nonlinear transformations on hyperspheres and argued for their ability to render data linearly separable through iterative "partitioning-conditional rotation". Finally, we connected ball-preserving transformations under multi-hyperplane partitioning with coupling layers in invertible neural networks, providing a profound geometric interpretation for the latter. Future work includes extending this framework to multi-layer deep architectures and multi-class problems, exploring analogous frameworks on more general manifolds, and designing new efficient and interpretable neural network modules based on this theory.

However, some tasks remain incomplete and require further research.

First, to provide a proof of arbitrary separability in the continuous case.

Second, to provide a proof of the approximation capability for the invertible module based on Householder reflections.

Third, to search for more general models of lossless information flow based on multi-hyperplane partitioning on hyperspheres.

Fourth, to explore whether other manifold structures, besides the hyperspherical model, also admit lossless, manifold-preserving nonlinear transformations.

## References

[1] Cover, T. M. (1965). Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition. *IEEE transactions on electronic computers*, (3), 326-334.

[2] Behrmann, J., Grathwohl, W., Chen, R. T., Duvenaud, D., & Jacobsen, J. H. (2019). Invertible residual networks. In *International Conference on Machine Learning* (pp. 573-582). PMLR.

[3] Kobyzev, I., Prince, S. J., & Brubaker, M. A. (2020). Normalizing flows: An introduction and review of current methods. *IEEE transactions on pattern analysis and machine intelligence*, 43(11), 3964-3979.

[4] Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). Density estimation using real nvp. *International Conference on Learning Representations*.

[5] Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative flow with invertible 1x1 convolutions. *Advances in neural information processing systems*, 31.

[6] Schölkopf, B., & Smola, A. J. (2002). *Learning with kernels: support vector machines, regularization, optimization, and beyond*. MIT press.

[7] Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. *arXiv preprint arXiv:2104.13478*.

[8] Arora, S., et al. (2019). Theory of deep learning. *Theoretical Foundations of Deep Learning*.

[Improvising the Learning of Neural Networks on Hyperspherical Manifold](https://zhuanzhi.ai/paper/9dca40de43029f58080fbc17ad7c29c4)

[On Learning Deep O(n)-Equivariant Hyperspheres](https://arxiv.org/abs/2305.15613)
