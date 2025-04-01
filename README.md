# Theoretical-Analysis-of-GraphArc

In this section, we provide a theoretical justification for the effectiveness of **GraphArc** in enhancing Out-of-Distribution (OOD) detection. Our analysis focuses on two key properties that have been shown to correlate strongly with OOD detection robustness: **intra-class variation** and **inter-class separation** (Ming et al., 2022). We demonstrate that GraphArc explicitly minimizes intra-class variation and enlarges inter-class separation through its **angular optimization** and **scaling design**.

---

### Reduced Intra-Class Variation via Angular Normalization

**Proposition 1 (Reduced Intra-Class Variation).**  
*Given normalized feature vectors and angular-based optimization, GraphArc minimizes intra-class variation under a cosine similarity metric.*

**Proof.**  
The normalized feature is given by  
$\tilde{h}_i^{(k)} = \frac{h_i^{(k)}}{||h_i^{(k)}||}$,  
and the normalized class center is  
$\tilde{W}_{y_i} = \frac{W_{y_i}}{||W_{y_i}||}$.

The angular softmax loss is defined as:  
$\mathcal{L}_{\text{angular}} = \frac{1}{N} \sum_{i} -\log \left( \frac{e^{s \cdot \cos(\theta_{y_i, i})}}{\sum_j e^{s \cdot \cos(\theta_{j, i})}} \right)$

This loss maximizes  
$\cos(\theta_{y_i, i}) = \tilde{W}_{y_i}^T \tilde{h}_i^{(k)}$,  
pulling same-class features toward the same angular direction. Since all class features are drawn toward a shared class anchor on the hypersphere, the angular spread within class $y_i$ is minimized:

$\mathbb{E}_{x_i, x_j \sim y_i} \left[ 1 - \cos\left( \angle(\tilde{h}_i^{(k)}, \tilde{h}_j^{(k)}) \right) \right] \to 0$

This reduction in intra-class angular variance improves feature compactness and the reliability of confidence calibration under distribution shift.

---

### Enhanced Inter-Class Separation via Scaling

**Proposition 2 (Increased Inter-Class Separation).**  
*Given a Lipschitz continuous feature mapping $\phi$, GraphArc increases the inter-class separation through its use of weight normalization and a scaling factor $s$.*

**Proof Sketch.**  
Consider the final layer representation of a GNN:  
$H^{(L)} = F^{(L)}\left(s \cdot \frac{W^{(L)}}{||W^{(L)}||} H^{(L-1)} Q\right)$

For two node embeddings $\phi(x_i), \phi(x_j)$ from different classes, their logits are:  
$z_i = s \cdot \cos(\theta_{y_i, i}), \quad z_j = s \cdot \cos(\theta_{y_j, j})$

Then the angular margin between them becomes:  
$||z_i - z_j|| = s \cdot |\cos(\theta_{y_i, i}) - \cos(\theta_{y_j, j})|$

As $s$ increases, the separation in logit space is amplified. Assuming $\phi$ is Lipschitz continuous with constant $L$, we have:  
$||\phi(x_i) - \phi(x_j)|| \leq L ||x_i - x_j|| \Rightarrow ||z_i - z_j|| \geq s \cdot L \cdot ||x_i - x_j||$

Thus, GraphArc provides a theoretical lower bound on the inter-class margin, proportional to the scaling factor $s$.

---

### Implications for OOD Detection

According to Ming et al. (2022), robust OOD detection is facilitated by minimizing intra-class variation $\mathcal{V}(\phi, \mathcal{E})$ and maximizing inter-class separation $\mathcal{I}_\rho(\phi, \mathcal{E})$. GraphArc satisfies both:

- ✅ **Normalization** aligns same-class features to a shared direction, minimizing $\mathcal{V}(\phi, \mathcal{E})$.
- ✅ **Scaling** widens angular margins and improves $\mathcal{I}_\rho(\phi, \mathcal{E})$.

These properties help GNNs better distinguish OOD samples and generate more calibrated confidence scores.
