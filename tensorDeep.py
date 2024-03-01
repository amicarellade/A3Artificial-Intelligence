from cleanData import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

file = "/Users/danteamicarella/Downloads/student-mat_modified (1)-1.csv"
df = preprocess_data(file)

columns_drop = ["Performance", "sex", "nursery"]

X = df.drop(columns_drop, axis = 1)
Y = df["Performance"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train the model
model = MLPClassifier(hidden_layer_sizes=(10, 8), activation='tanh', solver='adam', random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))





