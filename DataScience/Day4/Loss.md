# 🎯 Loss Functions in Deep Learning

Loss functions measure how well the model is performing by calculating the difference between predicted and actual values. Choosing the right loss function is crucial based on the task (regression, classification, etc.).

---

## 🔍 Loss Functions Comparison Table

| Loss Function              | Formula (Simplified)                       | Problem Type          | Pros                                                 | Cons                                                    | Use Cases                              |
|---------------------------|--------------------------------------------|------------------------|------------------------------------------------------|---------------------------------------------------------|----------------------------------------|
| **Mean Squared Error (MSE)**   | (1/n) Σ(y - ŷ)²                        | Regression             | Penalizes large errors, smooth gradient              | Sensitive to outliers                                   | Continuous output (e.g., price prediction) |
| **Mean Absolute Error (MAE)**  | (1/n) Σ|y - ŷ|                         | Regression             | Robust to outliers                                   | Non-smooth gradient                                     | Robust regression models               |
| **Huber Loss**                | Hybrid of MSE & MAE                     | Regression             | Combines benefits of MSE & MAE                       | Requires δ hyperparameter                              | Outlier-resistant regression            |
| **Log-Cosh Loss**            | log(cosh(y - ŷ))                        | Regression             | Smooth and less sensitive to outliers                | Slightly more complex                                   | Balanced regression problems           |
| **Binary Cross-Entropy**      | -[y log(ŷ) + (1-y) log(1-ŷ)]           | Binary Classification   | Probabilistic, well-suited for 0/1 tasks             | Can saturate with poor initialization                  | Binary classification (e.g., spam vs ham) |
| **Categorical Cross-Entropy** | -Σ yᵢ log(ŷᵢ)                         | Multi-Class Classification | Works well with one-hot labels                      | Labels must be one-hot encoded                        | Image classification, NLP               |
| **Sparse Categorical Cross-Entropy** | Like above, but with integer labels | Multi-Class Classification | No need for one-hot encoding                         | Labels must be integers                                | NLP, multi-class with sparse labels     |
| **Kullback-Leibler Divergence (KLDiv)** | Σ y log(y / ŷ)                | Distribution Comparison | Measures difference between two probability dists    | Asymmetric, sensitive to low probabilities             | Language modeling, GANs, distillation   |
| **Poisson Loss**             | ŷ - y * log(ŷ)                         | Count Regression        | Suitable for modeling count data                     | Assumes Poisson distribution                           | Insurance claims, count prediction      |
| **Hinge Loss**               | max(0, 1 - y·ŷ)                        | Binary Classification   | Used in SVMs, margin-based                          | Only for SVM-style models                             | SVM, binary classifiers with margins    |
| **Squared Hinge Loss**       | (max(0, 1 - y·ŷ))²                    | Binary Classification   | Stronger penalty than hinge                         | Same as above                                          | SVM variants                            |
| **Cosine Similarity Loss**   | 1 - cosine_similarity(y, ŷ)           | Embedding/Similarity    | Captures angle, not magnitude                       | Not useful for absolute values                         | Face recognition, recommendation        |
| **Triplet Loss**             | max(d(a,p) - d(a,n) + margin, 0)      | Embedding/Similarity    | Trains better embeddings                            | Hard to sample good triplets                          | FaceNet, Siamese networks               |
| **Contrastive Loss**         | y*d² + (1-y)*max(margin - d, 0)²      | Similarity Learning      | Works with pairs of inputs                          | Requires pair generation                              | Siamese networks                        |
| **CTC Loss**                 | Connectionist Temporal Classification | Sequence-to-sequence    | For variable-length output without alignment         | Complex, requires specific model output               | Speech recognition, OCR                 |
| **Dice Loss**                | 1 - (2 * intersection / union)        | Image Segmentation      | Works well with imbalanced classes                   | Non-differentiable parts                              | Medical image segmentation              |
| **Focal Loss**               | -α*(1-ŷ)^γ * log(ŷ)                    | Imbalanced Classification| Downweights easy examples                           | Hyperparameters (α, γ) sensitive                      | Object detection (e.g., RetinaNet)      |

---

## 📌 Summary

- 📈 **Regression**: Use **MSE**, **MAE**, **Huber**, or **Poisson**
- 🔢 **Binary Classification**: Use **Binary Cross-Entropy**, **Hinge**
- 🧠 **Multi-class**: Use **Categorical Cross-Entropy** or **Sparse**
- 🔍 **Similarity/Embeddings**: Use **Triplet**, **Contrastive**, **Cosine**
- 🧬 **Segmentation**: Use **Dice Loss**, **Focal Loss**
- 🔊 **Sequence/Temporal**: Use **CTC Loss**

---

## 🧠 Notes

- Use **Cross-Entropy** for most classification problems.
- Use **Huber Loss** if your regression data contains outliers.
- **Dice/Focal Loss** help with imbalanced image data.
- **Triplet/Contrastive** work in Siamese/embedding-based learning.

