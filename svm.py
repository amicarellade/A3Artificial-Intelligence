from cleanData import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import svm

file = "/Users/danteamicarella/Downloads/student-mat_modified (1)-1.csv"

df = preprocess_data(file)

# Feature Selection
# columns_drop = ["Performance", "higher", "famsup", "nursery", "school", "famrel", "sex"]
columns_drop = ["Performance"]

X = df.drop(columns_drop, axis = 1)
Y = df["Performance"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3)

clf = make_pipeline(StandardScaler(), SVC(kernel='sigmoid'))

# Train the model
clf.fit(X_train, Y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)