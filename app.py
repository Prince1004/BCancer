import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Load the CSV file
file_path = 'brestcancer.csv'
data = pd.read_csv(file_path)

# Handle categorical data
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# Specify the features you want to use
selected_features = [
    'radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean',
    'concave points_mean', 'radius_worst', 'perimeter_worst', 'area_worst',
    'concavity_worst', 'concave points_worst'
]

# Separate features and target
X = data[selected_features]  # Use only the selected features
y = data['diagnosis']  # Target

# Load the saved scaler and model
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

model = load_model('ann_model.h5')

# Sidebar for user inputs based on selected features
st.sidebar.header('User Input Parameters')

user_input = {}
for feature in selected_features:
    min_value = float(data[feature].min())
    max_value = float(data[feature].max())
    mean_value = float(data[feature].mean())
    user_input[feature] = st.sidebar.number_input(f'Input {feature}', min_value=min_value, max_value=max_value, value=mean_value)

# Create a DataFrame for the new data with only the selected features
user_input_df = pd.DataFrame([user_input])

# Ensure the DataFrame has the correct columns (selected features)
user_input_df = user_input_df[selected_features]

# Include all original features for the scaler
all_features = list(scaler.feature_names_in_)
full_input_df = pd.DataFrame(columns=all_features)

# Fill the DataFrame with zeros
full_input_df.loc[0] = 0

# Update the DataFrame with the user input data
for feature in selected_features:
    full_input_df.at[0, feature] = user_input[feature]

# Preprocess the user input data
user_input_scaled = scaler.transform(full_input_df)

# Select only the required features for prediction
user_input_scaled_selected = user_input_scaled[:, [all_features.index(feature) for feature in selected_features]]

# Make predictions
prediction = model.predict(user_input_scaled_selected)
prediction_class = (prediction > 0.5).astype(int)

# Display user input data and predictions
st.write("User Input Data")
st.write(user_input_df)

st.write(f"Prediction: {'Malignant' if prediction_class[0][0] == 1 else 'Benign'}")
