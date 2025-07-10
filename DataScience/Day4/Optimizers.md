# 🧠 Deep Learning Optimizers Comparison

This table provides a detailed comparison of popular deep learning optimizers, their properties, advantages, disadvantages, and use cases.

## 🚀 Optimizers Table

| Optimizer        | Adaptive LR | Momentum | Key Features / Notes                                                           | Pros                                                    | Cons                                                   | Use Cases                             |
|------------------|-------------|----------|----------------------------------------------------------------------------------|---------------------------------------------------------|--------------------------------------------------------|---------------------------------------|
| **SGD**          | ❌          | ❌       | Standard stochastic gradient descent                                             | Simple, good baseline                                   | Slow convergence, stuck in local minima                | Basic models, small datasets          |
| **SGD + Momentum** | ❌        | ✅       | Adds velocity to gradients to smooth updates                                     | Escapes local minima, faster than plain SGD             | Still may need tuning                                 | Image classification, CV models       |
| **Nesterov Momentum** | ❌    | ✅       | Improved momentum with a look-ahead gradient                                     | Faster convergence than SGD + momentum                  | Harder to tune                                        | Deep CNNs, better than SGD sometimes   |
| **Adagrad**       | ✅          | ❌       | Learning rate scales inversely with past gradient sum                            | Good for sparse data, adapts per-parameter learning rates | Learning rate shrinks too much                         | NLP, recommender systems              |
| **RMSprop**       | ✅          | ❌       | Uses moving average of squared gradients                                         | Fast, handles non-stationary objectives                 | Not ideal for convex problems                          | RNNs, time series, noisy data         |
| **Adadelta**      | ✅          | ✅       | Extension of Adagrad, avoids decaying learning rate                              | No manual learning rate needed, stable                  | Can be slow to converge                                | Deep networks with long training      |
| **Adam**          | ✅          | ✅       | Combines RMSprop + momentum, tracks 1st & 2nd moments                            | Fast, most commonly used, works well out-of-box         | Can overshoot minima, sensitive to tuning             | General deep learning, all tasks      |
| **Adamax**        | ✅          | ✅       | Variant of Adam using infinity norm instead of L2 norm                           | More stable in some models                              | Less commonly used                                    | Large parameter models                |
| **Nadam**         | ✅          | ✅       | Adam + Nesterov momentum                                                         | Faster convergence in some cases                        | Slightly more complex                                 | Deep networks                         |
| **FTRL**          | ✅          | ✅       | Follow-The-Regularized-Leader (used in large-scale models)                       | Good for online learning and sparse models              | Complex to implement                                  | Large-scale online models (e.g., ads) |
| **LAMB**          | ✅          | ✅       | Layer-wise Adaptive Moments optimizer (used in BERT, Transformers)               | Works well with large batch sizes                       | Requires large resources                               | Transformers, pre-trained models      |
| **AdaBound**      | ✅          | ✅       | Adam with dynamic bounds on learning rate                                        | Combines best of Adam & SGD                             | Less tested in real-world use                         | Stable training, adaptive + general   |
| **Yogi**          | ✅          | ✅       | Improved Adam with better control of update magnitudes                           | Avoids overshooting, stable updates                     | Newer, less widely adopted                            | NLP, long training models             |
| **AMSGrad**       | ✅          | ✅       | Variant of Adam to guarantee convergence                                         | Fixes non-convergence issue in Adam                     | Sometimes slower than Adam                            | Theoretical guarantees in training    |

## 🟢 Legend

- **Adaptive LR**: Adjusts learning rate for each parameter automatically.
- **Momentum**: Uses previous gradient direction to accelerate learning.
- ✅: Yes | ❌: No

## 🔥 Recommended Starting Points

- **Use `Adam`** for general-purpose training.
- **Use `RMSprop`** for RNNs or time-series data.
- **Use `SGD + Momentum`** for image-based or fine-tuning tasks.

---

🧠 *Tip:* Try multiple optimizers if training is unstable or too slow.
