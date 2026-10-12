import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==================================================
# IoT ANOMALY DETECTION SYSTEM
# ==================================================

# Step 1: Create project folders

os.makedirs("dataset", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 50)
print("IoT ANOMALY DETECTION SYSTEM")
print("=" * 50)

# ==================================================
# Step 2: Generate sample IoT network traffic data
# ==================================================

np.random.seed(42)

n_samples = 5000

data = pd.DataFrame({
    "packet_size": np.random.randint(40, 1500, n_samples),
    "packet_rate": np.random.randint(1, 1000, n_samples),
    "flow_duration": np.random.randint(1, 10000, n_samples),
    "failed_connections": np.random.randint(0, 20, n_samples)
})

# ==================================================
# Step 3: Label the network traffic
# 0 = Normal traffic
# 1 = Anomalous traffic
# ==================================================

data["label"] = (
    (data["packet_rate"] > 800)
    | (data["failed_connections"] > 15)
    | (
        (data["packet_rate"] > 650)
        & (data["packet_size"] < 100)
    )
).astype(int)

# Save the generated dataset

data.to_csv("dataset/iot_traffic.csv", index=False)

print("\nDataset created successfully!")
print("Total records:", len(data))
print("Normal records:", int((data["label"] == 0).sum()))
print("Anomalous records:", int((data["label"] == 1).sum()))

# ==================================================
# Step 4: Separate input features and target labels
# ==================================================

X = data.drop(columns=["label"])
y = data["label"]

# ==================================================
# Step 5: Split data into training and testing sets
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==================================================
# Step 6: Create the Random Forest model
# ==================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# ==================================================
# Step 7: Train the model
# ==================================================

print("\nTraining the machine learning model...")

model.fit(X_train, y_train)

print("Model training completed!")

# ==================================================
# Step 8: Evaluate the trained model
# ==================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )
)

# ==================================================
# Step 9: Save the trained model
# ==================================================

with open("models/anomaly_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nDataset saved: dataset/iot_traffic.csv")
print("Model saved: models/anomaly_model.pkl")

# ==================================================
# Training completed
# ==================================================

print("\nPROJECT TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 50)