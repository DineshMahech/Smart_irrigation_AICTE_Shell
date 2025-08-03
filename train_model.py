from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib

# Create dummy data with 20 features
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Train a simple RandomForest model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model to a .pkl file
joblib.dump(model, 'Farm_Irrigation_System.pkl')

print("Model saved successfully as Farm_Irrigation_System.pkl")
# The model is now ready to be used for predictions in the irrigation system.
# You can load this model later using joblib.load('Farm_Irrigation_System.pkl')
