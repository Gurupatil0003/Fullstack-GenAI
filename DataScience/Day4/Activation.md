# ⚡ Activation Functions in Deep Learning

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Here's a comparison of commonly used activation functions.

## 🔢 Activation Functions Comparison Table

| Activation Function | Equation (Simplified)       | Range           | Differentiable | Pros                                                    | Cons                                                   | Use Cases                              |
|---------------------|-----------------------------|------------------|----------------|---------------------------------------------------------|--------------------------------------------------------|----------------------------------------|
| **Sigmoid**         | 1 / (1 + e^(-x))            | (0, 1)           | ✅              | Smooth curve, good for probability                      | Vanishing gradient, slow convergence                   | Binary classification (output layer)   |
| **Tanh**            | (e^x - e^(-x)) / (e^x + e^(-x)) | (-1, 1)       | ✅              | Zero-centered, better than sigmoid                      | Vanishing gradient, slow learning                      | Hidden layers in shallow networks      |
| **ReLU**            | max(0, x)                   | [0, ∞)           | ✅              | Fast, simple, sparsity                                 | Dying ReLU problem                                     | CNNs, general hidden layers            |
| **Leaky ReLU**      | x if x > 0 else αx          | (-∞, ∞)          | ✅              | Solves dying ReLU, allows small gradients               | May still be unstable                                 | Deep networks with dying ReLU issue    |
| **Parametric ReLU** | x if x > 0 else a*x         | (-∞, ∞)          | ✅              | Learnable slope for negative inputs                    | Risk of overfitting if not regularized                 | Advanced deep networks                 |
| **ELU**             | x if x > 0 else α*(e^x - 1) | (-α, ∞)          | ✅              | Zero mean outputs, avoids dying neurons                 | Slower computation than ReLU                          | Complex networks, image tasks          |
| **SELU**            | λ * ELU(x)                  | (-∞, ∞)          | ✅              | Self-normalizing networks                              | Requires careful initialization and architecture       | Self-normalizing deep nets             |
| **Swish**           | x * sigmoid(x)              | (-0.28, ∞)       | ✅              | Smooth, non-monotonic, performs better in some models   | Slower to compute                                     | Transformers, deep CNNs, BERT          |
| **GELU**            | x * Φ(x)                    | (-∞, ∞)          | ✅              | Better than ReLU in some models, smooth                | Computation-intensive                                 | BERT, Transformers, NLP                |
| **Softmax**         | e^(xᵢ) / Σ e^(xⱼ)           | (0, 1) (sum=1)   | ✅              | Converts logits to probabilities                       | Not used in hidden layers                             | Multi-class classification (output)    |
| **Hard Sigmoid**    | Approx. linear around 0     | [0, 1]           | ✅              | Fast approximation of sigmoid                          | Less accurate, not smooth                             | Lightweight models, mobile devices     |
| **Hard Swish**      | x * ReLU6(x+3)/6            | ~(-0.25, ∞)      | ✅              | Fast & efficient Swish approximation                   | Less precise than Swish                               | MobileNetV3, edge devices              |

---

## 🧠 Summary

- **Use ReLU** for most hidden layers (fast & effective).
- **Use Softmax** in output layer for multi-class classification.
- **Use Sigmoid** in output layer for binary classification.
- **Use GELU / Swish** in Transformer-based models.
- **Use Leaky ReLU or ELU** to fix dying ReLU issues.

---

## 📌 Notes

- Smooth ≠ always better. Choose based on task and network architecture.
- Some activations (Swish, GELU) perform better with **large-scale models**.
- Always **normalize inputs** and use **proper weight initialization** to reduce vanishing gradient issues.

