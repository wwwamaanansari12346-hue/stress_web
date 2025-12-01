import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib
import numpy as np

def build_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def preprocess_data(df):
    drop_cols = ["student_id", "timestamp", "stress_level", "label_source"]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df["stress_level"] if "stress_level" in df.columns else None
    
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype == "object":
                X[col] = X[col].fillna(X[col].mode()[0])
            else:
                X[col] = X[col].fillna(X[col].mean())
    
    label_encoders = {}
    for col in X.select_dtypes(include="object").columns:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if y is not None:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        return X_scaled, y_categorical, scaler, label_encoders, le
    
    return X_scaled, None, scaler, label_encoders, None

def train_neural_network():
    df = pd.read_csv("complex_training_data.csv")
    print(f"Dataset size: {len(df)}")
    print("Class balance:\n", df["stress_level"].value_counts())
    
    drop_cols = ["student_id", "timestamp", "stress_level", "label_source"]
    features = [c for c in df.columns if c not in drop_cols]
    print("Features used:", features)
    
    X_scaled, y_categorical, scaler, label_encoders, label_encoder = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_categorical, test_size=0.2, random_state=42
    )
    
    model = build_model(X_train.shape[1], y_categorical.shape[1])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    print("\nModel Evaluation:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_labels, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred))
    print("Accuracy:", accuracy_score(y_test_labels, y_pred))
    
    model.save("g:\\MINOR PRJT FILE\\nn_stress_model.h5")
    joblib.dump(scaler, "g:\\MINOR PRJT FILE\\nn_scaler.joblib")
    joblib.dump(label_encoders, "g:\\MINOR PRJT FILE\\nn_label_encoders.joblib")
    joblib.dump(label_encoder, "g:\\MINOR PRJT FILE\\nn_target_encoder.joblib")

def predict_stress_level(input_dict):
    model = tf.keras.models.load_model("g:\\MINOR PRJT FILE\\nn_stress_model.h5")
    scaler = joblib.load("g:\\MINOR PRJT FILE\\nn_scaler.joblib")
    label_encoders = joblib.load("g:\\MINOR PRJT FILE\\nn_label_encoders.joblib")
    label_encoder = joblib.load("g:\\MINOR PRJT FILE\\nn_target_encoder.joblib")
    
    input_df = pd.DataFrame([input_dict])
    
    for col in input_df.columns:
        if pd.isnull(input_df[col]).any():
            if col in label_encoders:
                input_df[col] = input_df[col].fillna(input_df[col].mode()[0])
            else:
                input_df[col] = input_df[col].fillna(input_df[col].mean())
        
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])
    
    X_scaled = scaler.transform(input_df)
    
    pred_prob = model.predict(X_scaled)
    pred_class = np.argmax(pred_prob, axis=1)
    return label_encoder.inverse_transform(pred_class)[0]

if __name__ == "__main__":
    train_neural_network()
    
    df = pd.read_csv("g:\\MINOR PRJT FILE\\complex_training_data.csv")
    sample = df.iloc[0].to_dict()
    sample['gpa'] = None
    print("\nExample prediction with missing value:")
    print("Predicted stress level:", predict_stress_level(sample))