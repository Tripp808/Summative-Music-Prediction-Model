# Predicting Song Popularity: A Neural Network Approach

This project focuses on predicting the potential popularity of songs using a neural network model, based on various streaming metrics such as Spotify streams, YouTube views, TikTok views, and Shazam counts. The primary goal is to help music artists identify tracks with high potential for success, assisting in strategic decisions such as choosing tracks for touring and live performances. Although this approach differs from the original proposal i submitted due to the unavailability of specific datasets and limitations in public API access (such as privacy constraints of Spotify), it still addresses a significant challenge in the music industry—assessing song potential using streaming data.

**Note:** The performance metrics presented here are from the initial execution of the notebook. Due to Colab runtime disconnections, re-running the cells led to slightly different results. Nonetheless, the L2 regularized model consistently achieved superior performance across all runs. The L2 regularized model outperformed the baseline (vanilla) by approximately **3.44%** in accuracy.

## Dataset
Link to dataset: https://drive.google.com/file/d/1QUnDD9PeY6yAG1KgaDhgn700B0j5DVuK/view?usp=sharing
The dataset used in this project is rich with streaming-related metrics that are indicative of a song's popularity. The features such as:
- **Spotify Streams**
- **YouTube Views**
- **TikTok Views**
- **Shazam Counts**...

A binary target column, `High_Potential`, was based on a threshold in the "Track Score" column to classify songs as either having high potential (1) or low potential (0).

## Objectives

The key objectives of this project were:
1. To classify songs into two categories: high potential and low potential.
2. To optimize a neural network model to accurately predict the potential of a song based on streaming data.
3. To identify optimization techniques that can enhance model performance, ensuring a well-generalized model.

## Model Development and Implementation

The project employed three models to evaluate the impact of different regularization techniques on neural network performance:
1. **Vanilla Neural Network (Baseline)**
2. **L1 Regularized Neural Network**
3. **L2 Regularized Neural Network**

### Libraries Used

The implementation was carried out using Python and the following libraries:
- **TensorFlow and Keras**: For building and training the neural network models.
- **NumPy and Pandas**: For data manipulation and preprocessing.
- **Scikit-Learn**: For data splitting and metric evaluations.
- **Matplotlib and Seaborn**: For visualizing data and performance metrics.

### Model Training and Optimization Techniques

#### Vanilla Model (Baseline)

The vanilla neural network was used as a baseline for comparison:
- **Architecture**: The network consisted of an input layer, two hidden layers, and an output layer with a sigmoid activation function.
- **Loss Function**: Binary cross-entropy, suitable for binary classification.
- **Performance**:
  - Test Loss: **0.1350**
  - Accuracy: **94.89%**
  - Precision: **90.94%**
  - Recall: **100%**
  - F1 Score: **95.26%**

The vanilla model demonstrated decent performance but struggled with false positives and exhibited the highest test loss.

#### L1 Regularization

L1 regularization adds a penalty proportional to the absolute value of the weights, promoting sparsity in the model:
- **Penalty (Lambda)**: Set to 0.01 based on grid search results.
- **Relevance**: Encourages the model to zero out less important features, which can reduce overfitting.
- **Performance**:
  - Test Loss: **0.1021** (improved from the vanilla model)
  - Accuracy: **98.04%**
  - Precision: **98.92%** (highest precision among all models)
  - Recall: **97.25%**
  - F1 Score: **98.08%**

The L1 regularized model showed significant improvement over the vanilla model, especially in precision and overall accuracy.

#### L2 Regularization

L2 regularization adds a penalty proportional to the square of the weights, effectively reducing overfitting while ensuring a more generalizable model:
- **Penalty (Lambda)**: Set to **0.05** based on grid search results.
- **Relevance**: This technique ensures a smoother fit to the data and reduces overfitting, leading to a well-balanced model.
- **Performance**:
  - Test Loss: **0.0649** (lowest among all models)
  - Accuracy: **98.15%** (best performing model)
  - Precision: **98.30%**
  - Recall: **98.09%** (highest recall among the regularized models)
  - F1 Score: **98.20%** (highest overall)

The L2 regularized model outperformed the other models in terms of accuracy, recall, and F1 score, proving to be the best fit for the project.

### Parameter Tuning and Optimization Techniques

#### Learning Rate Scheduling

- **Principle**: Dynamically adjusts the learning rate to stabilize training and improve convergence.
- **Implementation**: The learning rate was set to start at **0.01**, with a reduction factor of **0.5** every **10 epochs**, based on empirical observations.
- **Relevance**: Learning rate scheduling ensured efficient training by allowing rapid learning during early epochs and fine-tuning as training progressed. This helped improve convergence speed and model accuracy across all model variations.

#### Dropout Regularization

- **Principle**: Reduces overfitting by randomly dropping neurons during training.
- **Implementation**: Dropout was applied with varying rates across models:
  - **Vanilla Model**: Dropout was not applied, serving as a baseline for comparison.
  - **L1 Model**: Dropout rates of **0.5** for the first layer and **0.2** for subsequent layers were used.
  - **L2 Model**: Dropout rates of **0.3** for the first layer and **0.2** for the next layers were implemented.
- **Relevance**: Dropout regularization helped to improve the generalization capabilities of the models by preventing overfitting, especially in the L1 and L2 regularized versions.

#### Early Stopping

- **Principle**: Prevents overfitting by stopping the training process when the model's performance on a validation set no longer improves.
- **Implementation**: Early stopping was applied with a patience value of **10 epochs** for all models (Vanilla, L1, and L2), where training was halted if there was no improvement in validation loss over this period.
- **Relevance**: This technique allowed the models to maintain generalization to unseen data by avoiding excessive training beyond the optimal point. It ensured efficient training and prevented overfitting, leading to more robust model performance.

#### Optimizers

- **Testing and Selection**: Both **RMSprop** and **Adam** optimizers were tested for the different models. After evaluating their performance, the best combinations were selected for final implementation.
  - **L1 Regularized Model**: Compiled using the **Adam** optimizer, known for its efficiency and performance in handling sparse gradients, making it suitable for this model's architecture.
  - **L2 Regularized Model**: Also utilized the **RMSprop** optimizer to enhance training stability and performance.
- **Relevance**: The choice of optimizers contributed to the models' ability to converge effectively during training. The RMSprop optimizer proved particularly beneficial for the L2 model, while the Adam optimizer provided robustness for the L1 regularized version.

### Key Observations and Performance Analysis

| Model              | Test Loss | Accuracy | Precision | Recall | F1 Score |
|--------------------|-----------|----------|-----------|--------|----------|
| **Vanilla**        | 0.1350    | 94.89%   | 90.94%    | 100%   | 95.26%   |
| **L1 Regularized** | 0.1021    | 98.04%   | 98.92%    | 97.25% | 98.08%   |
| **L2 Regularized** | 0.0649    | 98.15%   | 98.30%    | 98.09% | 98.20%   |

The L2 regularized model showed an approximately **3.44% improvement in accuracy** over the vanilla model, making it the most robust choice for predicting song popularity.

## Running the Notebook and Loading the Saved Models

To run the notebook and load the saved models:
**Clone the repository** and navigate to the project directory:
   ```bash
   1. git clone <https://github.com/Tripp808/Summative-Music-Prediction-Model>
   2. cd <Song_Popularity_Prediction_Model> 
   3. Install the required dependencies using pip install
   4. Then run the notebook jupyter notebook Oche_Ankeli_notebook.ipynb
   5. Run the cells in sequence to preprocess the data, train the models, and evaluate performance.
   6. Load the pre-trained models by using the load_model() function from Keras, specifying the path to the saved model files.
```
    
## Conclusion
The L2 regularization technique with the RMSprop optimizer proved to be the most effective solution, achieving the best balance across all performance metrics. Although L1 regularization showed high precision, it did not perform as well in terms of recall as the L2 model. The vanilla model lagged behind, indicating that regularization techniques were essential for improving generalization and minimizing overfitting.

The findings highlight the potential of leveraging streaming metrics to predict the popularity of songs, providing valuable insights for artists and music industry stakeholders.

Note: The performance metrics presented are from the initial execution of the notebook. Due to Colab runtime disconnections, re-running the cells led to slightly different results. However, the L2 regularized model consistently outperformed the others across multiple runs.

