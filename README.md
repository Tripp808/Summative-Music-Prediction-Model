# Predicting Song Popularity: A Neural Network Approach
This project focuses on predicting the potential popularity of songs using a neural network model based on various streaming metrics. The goal is to help music artists identify tracks with high potential for success and determining which is best for touring and live performances by classifying songs into two categories: high potential and low potential, based on streaming-related data.
Although this approach differs from the original proposal i submitted due to the unavailability of certain datasets and API limitations (such as Spotify's privacy constraints), it still addresses a key challenge within the music industry.

**Note**: The performance metrics presented here are from the initial notebook execution. Due to Colab runtime disconnections, re-running the cells led to slightly different results. Nonetheless, the L2 regularized model consistently achieved superior performance in both runs. 
The L2 regularized model outperformed the baseline by approximately **3.44%** in accuracy.

## Model Development and Optimization Techniques

### Overview of Models and Techniques
The models used include a vanilla neural network, an L1 regularized model, and an L2 regularized model. The following optimization techniques were applied to improve performance:

1. **Vanilla Model**: This served as the baseline for evaluating the effectiveness of the regularization techniques.
   - **Test Loss**: 0.1350 (highest among all models)
   - **Accuracy**: 94.89% (lowest)
   - **Precision**: 90.94%
   - **Recall**: 100%
   - **F1 Score**: 95.26%

2. **L1 Regularization**
   - **Principle**: Adds a penalty proportional to the absolute value of weights, encouraging sparsity.
   - **Test Loss**: 0.1021 (better than vanilla)
   - **Accuracy**: 98.04% (significantly higher than vanilla)
   - **Precision**: 98.92% (highest)
   - **Recall**: 97.25%
   - **F1 Score**: 98.08%

3. **L2 Regularization**
   - **Principle**: Adds a penalty proportional to the squared value of weights, reducing overfitting.
   - **Relevance**: Ensured a better fit while maintaining model generalization.
   - **Parameter Tuning**: Lambda was set to 0.05 based on grid search results.
   - **Test Loss**: 0.0649 (lowest)
   - **Accuracy**: 98.15% (best performing model)
   - **Precision**: 98.30%
   - **Recall**: 98.09% (highest among regularized models)
   - **F1 Score**: 98.20% (highest)

### Learning Rate Scheduling
   - **Principle**: Dynamically adjusts learning rate during training.
   - **Parameter Tuning**: Started at 0.01 with a reduction factor of 0.5 every 10 epochs.

### Dropout
   - **Principle**: Reduces overfitting by randomly dropping neurons.
   - **Implementation**: Set to 0.3 based on empirical testing.

### Batch Normalization
   - **Principle**: Normalizes inputs to each layer, stabilizing the training process.
   - **Implementation**: Applied after hidden layers with momentum set to 0.99.

## Key Observations and Performance Analysis

| Model                  | Test Loss | Accuracy | Precision | Recall | F1 Score |
|------------------------|-----------|----------|-----------|--------|----------|
| **Vanilla**            | 0.1350    | 94.89%   | 90.94%    | 100%   | 95.26%   |
| **L1 Regularized**     | 0.1021    | 98.04%   | 98.92%    | 97.25% | 98.08%   |
| **L2 Regularized**     | 0.0649    | 98.15%   | 98.30%    | 98.09% | 98.20%   |

The L2 regularized model outperformed the baseline by approximately **3.44%** in accuracy.

### Conclusion
L2 regularization with RMSprop proved to be the best fit, demonstrating superior performance across all metrics. The L1 model, though strong in precision, lagged in recall. The vanilla model showed decent results but struggled with false positives and higher test loss.
