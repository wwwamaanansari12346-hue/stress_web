def preprocess_input_data(input_dict, label_encoders, scaler):
    input_df = pd.DataFrame([input_dict])
    
    # Handle missing values and encode categorical variables
    for col in input_df.columns:
        if pd.isnull(input_df[col]).any():
            if col in label_encoders:
                input_df[col] = input_df[col].fillna(input_df[col].mode()[0])
            else:
                input_df[col] = input_df[col].fillna(input_df[col].mean())
        
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])
    
    # Scale input
    X_scaled = scaler.transform(input_df)
    
    return X_scaled

def validate_input_data(input_dict, expected_columns):
    missing_cols = [col for col in expected_columns if col not in input_dict]
    if missing_cols:
        raise ValueError(f"Missing input data for columns: {', '.join(missing_cols)}")