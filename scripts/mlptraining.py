import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import datetime


def main():
    # 1. Define file paths
    # Input CSV file containing pre-extracted MidiBERT features
    csv_file = r'E:\Master\symbolic_2026\features\filtered_midibert_768d_features.csv'
    # Output text file to save evaluation results
    output_file = 'mlp_results.txt'

    print(f"Loading MidiBERT pre-trained feature data from: {csv_file}...")
    df = pd.read_csv(csv_file)

    # 2. Separate features (X) and labels (y)
    # Columns to exclude from features (metadata columns)
    metadata_cols = ['filename', 'clean_name', 'composer']

    # Drop metadata columns to get the feature matrix X
    # Only drop columns that actually exist in the dataframe to avoid errors
    X = df.drop(columns=[col for col in metadata_cols if col in df.columns])

    # Extract the target variable ('composer') and encode it into integers
    y_raw = df['composer']
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"Feature dimensions: {X.shape[1]}, Total samples: {X.shape[0]}")
    print("Splitting dataset and applying standardization...")

    # 3. Split dataset and standardize features
    # Split into training and testing sets (80% train, 20% test)
    # stratify=y ensures the class distribution is preserved in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize StandardScaler and fit on training data, then transform both sets
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Initialize and train the MLP model
    print("Training MLP model, please wait...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),  # Two hidden layers with 128 and 64 neurons
        activation='relu',  # ReLU activation function
        solver='adam',  # Adam optimizer
        alpha=0.01,  # L2 regularization parameter
        max_iter=1000,  # Maximum number of iterations
        random_state=42,  # For reproducibility
        early_stopping=True,  # Enable early stopping to prevent overfitting
        validation_fraction=0.1  # Fraction of training data to use as validation set
    )

    # Train the model
    mlp.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = mlp.predict(X_test_scaled)

    # 5. Generate evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    # Generate detailed classification report (precision, recall, f1-score)
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    # Get current timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 6. Construct and save the results to a text file
    result_text = (
        f"========== MidiBERT (768D) MLP Classification Results ==========\n"
        f"Run Time: {current_time}\n"
        f"Feature Dimensions: {X.shape[1]}\n"
        f"Total Samples: {X.shape[0]}\n"
        f"Model Parameters: MLPClassifier(hidden_layer_sizes={mlp.hidden_layer_sizes}, alpha={mlp.alpha})\n"
        f"--------------------------------------------------\n"
        f"Overall Accuracy: {acc:.4f}\n\n"
        f"Detailed Classification Report:\n"
        f"{report}\n"
        f"==================================================\n"
    )

    # Write results to file with UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result_text)

    print(f"✅ Model training and evaluation completed!")
    print(f"✅ Results successfully saved to: {output_file}")


if __name__ == "__main__":
    main()