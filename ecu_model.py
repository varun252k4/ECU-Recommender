import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import pickle

# Load the dataset
def load_data(file_path="ecu_cluster_dataset.csv"):
    df = pd.read_csv(file_path)
    return df

# Preprocess the data
def preprocess_data(df):
    ecu_ids = df["ECU_ID"]
    operational_state = df["Operational_State"]
    drop_columns = ["ECU_ID", "Operational_State"]

    categorical_cols = [
        "ECU_Type", "Protocol", "Redundancy",
        "Manufacturer", "Software_Version"
    ]

    numerical_cols = [
        "CPU_Speed_MHz", "Memory_MB", "Power_Watts", "Message_Frequency_msgs",
        "Unique_Message_IDs", "Error_Rate_percent", "Response_Time_ms",
        "Network_Topology_Level", "Inter_ECU_Dependencies",
        "Mean_Voltage_V", "Max_Voltage_V", "Bit_Time_us", "Plateau_Time_us"
    ]
    
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                              ("encoder", OneHotEncoder(handle_unknown="ignore"))])

    numerical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")),
                                            ("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(transformers=[("num", numerical_transformer, numerical_cols),
                                                   ("cat", categorical_transformer, categorical_cols)])

    # Apply the transformations
    X = preprocessor.fit_transform(df.drop(columns=drop_columns))
    return X, ecu_ids, operational_state, preprocessor



# Train KNN model
def train_knn_model(X, n_neighbors=5):
    knn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    knn_model.fit(X)
    return knn_model

# Save model and preprocessor for later use
def save_model(knn_model, preprocessor, model_filename="knn_model.pkl"):
    with open(model_filename, 'wb') as file:
        pickle.dump((knn_model, preprocessor), file)
    print(f"Model saved successfully to {model_filename}!")

# Main function to train and save the model
def train_and_save_model():
    # Load the dataset
    df = load_data("ecu_cluster_dataset.csv")
    # Preprocess the data
    X, ecu_ids, operational_state, preprocessor = preprocess_data(df)
    # Train the KNN model
    knn_model = train_knn_model(X)
    # Save the trained model and preprocessor
    save_model(knn_model, preprocessor)

if __name__ == "__main__":
    # Call the train_and_save_model function to train and save the model
    train_and_save_model()
