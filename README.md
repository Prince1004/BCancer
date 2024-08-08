# Brest Cancer
Cancer Prediction





Dataset Acquisition and Preparation
1. Download the Dataset:
DB link : https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
Attached the DB in git

cd BreastCancerAnalysis

2. Create and Activate a Virtual Environment
python -m venv env

#On Windows use env\Scripts\activate to activate

3. Install Dependencies
pip install -r requirements.txt

4. Download Pre-trained Model and Artifacts
Ensure the following files are in the project directory:

ann_model.h5 (The trained ANN model)

kbest.pkl (The pickled feature selector)

scaler.pkl (The pickled scaler)

If these files are not present, you may need to train the model and save these artifacts.

Deployment 

1. Start the Streamlit App
streamlit run app.py

2. Navigate to the Web App
 https://bcancer-rfq6h9ui3rbbo6ep9t9mbu.streamlit.app/

  
General Overview of brestcancer.ipynb
1. Data Preparation:
Dropped Unwanted Columns and Encoded Categorical features
Standardized the data



2: Feature Selection
Used Seleck K best and RFE Methods to find best 10 features


3:Model Tuning
Perofmed Grid Search Cross-Validation, RandomizedSearchCV 


4: Implementing an Artificial Neural Network (ANN) Model
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
