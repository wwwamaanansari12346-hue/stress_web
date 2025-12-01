# Stress Management System

This project is a Stress Management System that utilizes a neural network model to predict stress levels based on various input features. The system is designed to provide insights into stress management and support users in understanding their stress levels.

## Project Structure

The project is organized into several directories and files:

- **API/**: Contains the FastAPI application and related files for handling API requests.
  - **main.py**: Entry point for the API, initializes the FastAPI application.
  - **routers/**: Contains route definitions for the API.
    - **predict.py**: Defines the prediction endpoint for stress level predictions.
    - **health.py**: Provides a health check endpoint.
  - **services/**: Contains business logic for the application.
    - **prediction_service.py**: Logic for making predictions using the neural network model.
  - **models/**: Defines data models for request and response validation.
    - **schemas.py**: Contains schemas for validating API requests and responses.
  - **utils/**: Utility functions for preprocessing input data.
    - **preprocessing.py**: Functions for data preprocessing before predictions.
  - **deps.py**: Dependency injection functions for shared resources.
  - **Dockerfile**: Used to create a Docker image for the API.

- **models/**: Contains the trained machine learning models and preprocessing objects.
  - **nn_stress_model.h5**: Saved neural network model.
  - **nn_scaler.joblib**: Scaler object for feature scaling.
  - **nn_label_encoders.joblib**: Label encoders for categorical variables.

- **src/**: Contains the source code for the neural network model.
  - **mental_model(NN).py**: Implementation of the neural network model, including training and evaluation.

- **tests/**: Contains unit tests for the API.
  - **test_predict_api.py**: Tests for the prediction API.

- **requirements.txt**: Lists the dependencies required for the project.

- **.env.example**: Example environment variables for the project.

## Setup Instructions

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using pip:
   ```
   pip install -r requirements.txt
   ```
4. Set up environment variables as needed, using the `.env.example` as a reference.
5. Run the API using the following command:
   ```
   uvicorn API.main:app --reload
   ```

## Usage

Once the API is running, you can access the following endpoints:

- **Health Check**: `GET /health`
  - Returns a simple message indicating that the API is running.

- **Predict Stress Level**: `POST /predict`
  - Sends a JSON payload with the necessary input features to receive a predicted stress level.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.