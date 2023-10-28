from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('synthetic_dataset.csv')

# Handle categorical variables
le = LabelEncoder()
df['seating_type'] = le.fit_transform(df['seating_type'])

# Split data into train and test sets
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

# Define features and target
features = ['height_of_video_wall', 'width_of_video_wall', 'room_length', 'room_width', 'room_height', 'seating_type', 'number_of_seats']

# Define a function to train and evaluate a model
def train_evaluate(train_data, test_data, features, target):
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    
    # Initialize and train the model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    print(f'Root Mean Squared Error for {target}: {rmse:.2f}')
    
    return model

# Train model for each target
distance_model = train_evaluate(train_set, test_set, features, 'distance_from_video_wall')

row_data = df[df['seating_type'] == le.transform(['row'])[0]]
train_set_row, test_set_row = train_test_split(row_data, test_size=0.2, random_state=42)
seats_per_row_model = train_evaluate(train_set_row, test_set_row, features, 'seats_per_row')
number_of_rows_model = train_evaluate(train_set_row, test_set_row, features, 'number_of_rows')

cluster_data = df[df['seating_type'] == le.transform(['cluster'])[0]]
train_set_cluster, test_set_cluster = train_test_split(cluster_data, test_size=0.2, random_state=42)
seats_per_cluster_model = train_evaluate(train_set_cluster, test_set_cluster, features, 'seats_per_cluster')
number_of_clusters_model = train_evaluate(train_set_cluster, test_set_cluster, features, 'number_of_clusters')
from sklearn.metrics import mean_absolute_error, r2_score

# Define a function to compute and display metrics
def compute_metrics(model, X_test, y_test, target_name):
    y_pred = model.predict(X_test)
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Metrics for {target_name}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.2f}\n")
    
# Check metrics for each model
X_test = test_set[features]
compute_metrics(distance_model, X_test, test_set['distance_from_video_wall'], 'Distance from Video Wall')

X_test_row = test_set_row[features]
compute_metrics(seats_per_row_model, X_test_row, test_set_row['seats_per_row'], 'Seats per Row')
compute_metrics(number_of_rows_model, X_test_row, test_set_row['number_of_rows'], 'Number of Rows')

X_test_cluster = test_set_cluster[features]
compute_metrics(seats_per_cluster_model, X_test_cluster, test_set_cluster['seats_per_cluster'], 'Seats per Cluster')
compute_metrics(number_of_clusters_model, X_test_cluster, test_set_cluster['number_of_clusters'], 'Number of Clusters')
def predict_outputs(sample_details):
    # Convert categorical features to numeric (assuming you've previously used LabelEncoder named 'le')
    sample_details['seating_type'] = le.transform([sample_details['seating_type']])[0]
    
    # Convert the input details into a DataFrame
    sample_df = pd.DataFrame([sample_details])
    
    # Extract the features
    features_sample = sample_df[features]
    
    # Predict distance from video wall
    distance_prediction = distance_model.predict(features_sample)[0]
    
    if sample_details['seating_type'] == le.transform(['row'])[0]:  # If seating type is 'row'
        seats_per_row_prediction = seats_per_row_model.predict(features_sample)[0]
        number_of_rows_prediction = number_of_rows_model.predict(features_sample)[0]
        return {
            'distance_from_video_wall': distance_prediction,
            'seats_per_row': seats_per_row_prediction,
            'number_of_rows': number_of_rows_prediction
        }
    
    elif sample_details['seating_type'] == le.transform(['cluster'])[0]:  # If seating type is 'cluster'
        seats_per_cluster_prediction = seats_per_cluster_model.predict(features_sample)[0]
        number_of_clusters_prediction = number_of_clusters_model.predict(features_sample)[0]
        return {
            'distance_from_video_wall': distance_prediction,
            'seats_per_cluster': seats_per_cluster_prediction,
            'number_of_clusters': number_of_clusters_prediction
        }

# Example of how to use the function:
sample_details = {
    'height_of_video_wall': 3.5,
    'width_of_video_wall': 4.5,
    'room_length': 15,
    'room_width': 10,
    'room_height': 3.5,
    'seating_type': 'cluster',
    'number_of_seats': 30
}

output = predict_outputs(sample_details)
print(output)  

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Use the function you've provided to get predictions
    predictions = predict_outputs(data)
    
    # Return predictions as JSON response
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
