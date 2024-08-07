# BCancer
Cancer Prediction
Dataset Acquisition and Preparation
1. Download the Dataset:
DB link : https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
Attached the DB in git


3. Data Preparation:
Dropped Unwanted Columns and Encoded Categorical features
Standardized the data



Step 3: Feature Selection
Used Seleck K best and RFE Methods to find best 10 features


Step 4:Model Tuning
Perofmed Grid Search Cross-Validation, RandomizedSearchCV 


Step 5: Implementing an Artificial Neural Network (ANN) Model
Model Architecture
Input Layer:
Dense Layer: Initialized with the number of neurons specified by hidden_layer_sizes[0] and an activation function determined by the best parameters.
Batch Normalization: Stabilizes and accelerates training.
Dropout: Set at 30% to prevent overfitting by randomly ignoring neurons during training.
Hidden Layers:

Multiple dense layers, each followed by batch normalization and dropout, with neuron sizes specified by hidden_layer_sizes.
Output Layer:

Dense Layer: A single neuron with a sigmoid activation function, producing a probability for binary classification.
Model Compilation
The model is compiled using:

Optimizer: Adam, based on the best parameters.
Loss Function: Binary cross-entropy, ideal for binary classification tasks.
Metrics: Accuracy, to evaluate model performance.
Training the Model
The model is trained with:
Early Stopping: Monitors validation loss and stops training if no improvement is observed for ten epochs, preventing overfitting.
Validation Split: 20% of the training data is used for validation.
Epochs and Batch Size: The model trains for up to 100 epochs with a batch size of 32.
Evaluation
After training, the model's performance is evaluated on both training and test datasets:
Training Accuracy: Indicates how well the model learned from the training data.
Testing Accuracy: Measures how well the model generalizes to new, unseen data.
Results:

Training Accuracy: 96.04%
Testing Accuracy: 96.49%
Conclusion
This ANN model demonstrates effective binary classification through a well-structured architecture, careful compilation, and strategic training, achieving high accuracy on both training and test datasets.



Stream lit appliction output
![image](https://github.com/user-attachments/assets/cf8e6d90-f55f-4c81-82f7-6a8e2a98b021)
