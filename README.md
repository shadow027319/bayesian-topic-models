# Bayesian Topic Models

Bayesian inference over discrete text data: from maximum likelihood multinomials through to collapsed Gibbs sampling for Latent Dirichlet Allocation. Built for Cambridge Engineering Part IIB, 4F13 Machine Learning.

Corpus: ~3000 blog posts from [Daily Kos](http://www.dailykos.com), split into training set $A$ and held-out test set $B$, with a vocabulary of $W = 6906$ unique words.

---

## Models

### 1. Multinomial with Dirichlet Prior

The simplest bag-of-words model. Each document is drawn i.i.d. from a single multinomial over the vocabulary.

**Likelihood** (categorical form, ignoring ordering):

$$p(\mathbf{x} \mid \boldsymbol{\pi}) = \prod_{i=1}^{W} \pi_i^{x_i}$$

**Conjugate Dirichlet prior** with symmetric concentration $\alpha$:

$$p(\boldsymbol{\pi}) = \text{Dir}(\boldsymbol{\alpha}) \propto \prod_{i=1}^{W} \pi_i^{\alpha - 1}$$

**Posterior:**

$$p(\boldsymbol{\pi} \mid \mathbf{x}) = \text{Dir}(\mathbf{x} + \boldsymbol{\alpha})$$

**Predictive:**

$$p(x = i \mid \mathbf{x}) = \mathbb{E}[\pi_i] = \frac{x_i + \alpha}{n + W\alpha}$$

For rare words ($x_i < n/W$), the prior increases probability mass relative to ML; for common words it shrinks it. The Dirichlet acts as a smoother, pulling the distribution toward uniform. Setting $\alpha = 0$ recovers ML; $\alpha = 1$ gives a uniform (non-informative) prior.

**Bayesian predictive (integrating out $\boldsymbol{\pi}$):**

Rather than plug in a point estimate, we can integrate out $\boldsymbol{\pi}$ analytically. For a test document with word counts $\mathbf{c}$ and posterior pseudocounts $\mathbf{c}_0 = \mathbf{x}_{\text{train}} + \boldsymbol{\alpha}$:

$$\log p(\mathbf{c} \mid \mathbf{c}_0) = \log B(\mathbf{c} + \mathbf{c}_0) - \log B(\mathbf{c}_0)$$

where $B(\cdot)$ is the multivariate Beta function: $\log B(\mathbf{c}) = \sum_i \log \Gamma(c_i) - \log \Gamma(\sum_i c_i)$.

**Per-word perplexity:**

$$\text{PPL} = \exp\left(-\frac{\log p(\mathbf{c})}{n_d}\right)$$

where $n_d = \sum_i c_i$ is the document length. Lower is better. A uniform multinomial gives $\text{PPL} = W$.

### 2. Bayesian Mixture of Multinomials (BMM)

Each document belongs to one of $K$ clusters, each with its own word distribution. The generative process:

$$z_d \sim \text{Cat}(\boldsymbol{\theta}), \qquad \mathbf{x}_d \mid z_d = k \sim \text{Mult}(\boldsymbol{\phi}_k)$$

with Dirichlet priors on both:

$$\boldsymbol{\theta} \sim \text{Dir}(\alpha), \qquad \boldsymbol{\phi}_k \sim \text{Dir}(\gamma)$$

**Collapsed Gibbs sampling** integrates out $\boldsymbol{\theta}$ and $\boldsymbol{\phi}_k$, sampling only the cluster assignments $z_d$. The conditional for reassigning document $d$:

$$p(z_d = k \mid \mathbf{z}_{-d}, \mathbf{x}) \propto \left(n_k^{-d} + \alpha\right) \prod_{w} \frac{\Gamma(n_{wk}^{-d} + c_{wd} + \gamma)}{\Gamma(n_{wk}^{-d} + \gamma)}$$

where $n_k^{-d}$ is the number of documents assigned to component $k$ excluding $d$, and $n_{wk}^{-d}$ is the count of word $w$ in component $k$ excluding $d$.

In practice, the log-conditional is computed as:

$$\log p(z_d = k) = \log(n_k^{-d} + \alpha) + \sum_{w \in d} c_{wd} \left[\log(n_{wk}^{-d} + \gamma) - \log(s_k^{-d} + W\gamma)\right]$$

where $s_k = \sum_w n_{wk}$ is the total word count in component $k$.

**Test perplexity** is computed by marginalising over components:

$$p(\mathbf{x}_{\text{test}}) = \sum_{k=1}^{K} \theta_k \prod_{w} \phi_{kw}^{c_w}$$

using the smoothed estimates $\theta_k \propto n_k + \alpha$ and $\phi_{kw} = (n_{wk} + \gamma) / (s_k + W\gamma)$.

### 3. Latent Dirichlet Allocation (LDA)

Unlike BMM, LDA assigns topics *per word*, not per document. Each document has its own topic mixture.

**Generative process:**

$$\boldsymbol{\theta}_d \sim \text{Dir}(\alpha), \qquad z_{dn} \mid \boldsymbol{\theta}_d \sim \text{Cat}(\boldsymbol{\theta}_d), \qquad x_{dn} \mid z_{dn} = k \sim \text{Cat}(\boldsymbol{\phi}_k)$$

$$\boldsymbol{\phi}_k \sim \text{Dir}(\gamma)$$

**Collapsed Gibbs** integrates out both $\boldsymbol{\theta}_d$ and $\boldsymbol{\phi}_k$, sampling each word's topic assignment:

$$p(z_{dn} = k \mid \mathbf{z}_{-dn}, \mathbf{x}) \propto \left(\alpha + n_{dk}^{-dn}\right) \cdot \frac{\gamma + n_{wk}^{-dn}}{W\gamma + s_k^{-dn}}$$

where $n_{dk}$ is the number of words in document $d$ assigned to topic $k$, $n_{wk}$ is the number of times word $w$ is assigned to topic $k$ globally, and $s_k = \sum_w n_{wk}$.

**Predictive distribution** for held-out documents: run Gibbs on the test document with the global word-topic counts $n_{wk}$ and $s_k$ held fixed from training. Then:

$$p(w \mid d) = \sum_{k=1}^{K} \underbrace{\frac{\alpha + n_{dk}}{\sum_{k'} (\alpha + n_{dk'})}}_{\hat{\theta}_{dk}} \cdot \underbrace{\frac{\gamma + n_{wk}}{W\gamma + s_k}}_{\hat{\phi}_{kw}}$$

**Topic quality** is assessed via per-topic word entropy:

$$H_k = -\sum_{w=1}^{W} \hat{\phi}_{kw} \log_2 \hat{\phi}_{kw}$$

Lower entropy means a more concentrated, interpretable topic. As Gibbs sweeps progress, topics sharpen and entropy decreases from near-uniform ($\log_2 W \approx 12.75$ bits) to structured distributions.

---

## Structure

```
.
├── multinomial_inference.ipynb   # ML multinomial, Dirichlet prior, perplexity (parts a–c)
├── bmm_gibbs.ipynb               # Mixture of multinomials, Gibbs convergence (part d)
├── lda_gibbs.ipynb               # LDA, topic entropy, perplexity comparison (part e)
├── utils.py                      # data loading, plotting helpers
├── kos_doc_data.mat              # Daily Kos corpus (training A, test B, vocabulary V)
└── report.pdf                    # writeup
```

## Key Results

| Model | Test Perplexity |
|---|---|
| ML Multinomial ($\alpha = 0$) | undefined for unseen words |
| Dirichlet-smoothed ($\alpha = 100$) | ~2100 |
| Bayesian predictive (integrated) | ~2050 |
| BMM ($K=20$) | ~2100 |
| LDA ($K=20$) | ~1640 |
| Uniform baseline | 6906 |

LDA significantly outperforms all single-multinomial baselines. Per-word topic assignments capture within-document topical variation that BMM's document-level clustering cannot.

## Dependencies

```
numpy, scipy, matplotlib
```

## Context

Part of the Cambridge Engineering Tripos (Information & Computer Engineering), covering Bayesian nonparametric text modelling and MCMC inference. 
