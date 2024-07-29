# Databricks notebook source
# MAGIC %md
# MAGIC **LOAD THE TRAIN AND TEST DATA**

# COMMAND ----------



from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("example").getOrCreate()

# Access the global temporary views
X_train_spark = spark.sql("SELECT * FROM global_temp.table_X_train_spark")
X_test_spark = spark.sql("SELECT * FROM global_temp.table_X_test_spark")
y_train_spark = spark.sql("SELECT * FROM global_temp.table_y_train_spark")
y_test_spark = spark.sql("SELECT * FROM global_temp.table_y_test_spark")

# Show the data
X_train = X_train_spark.toPandas()
X_test = X_test_spark.toPandas()
y_train = y_train_spark.toPandas()
y_test = y_test_spark.toPandas()

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier


# # hyperparameter tuning
# param_grid = {
#     'n_estimators': [100 , 500],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [5, 10],
#     'min_samples_leaf': [2, 4],
#     'bootstrap': [True, False]
# }

# rf_model = RandomForestClassifier(random_state=42)
# grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)

# Initialize the Random Forest classifier
model = AdaBoostClassifier(n_estimators=100, random_state=42)
# model = RandomForestClassifier(n_estimators=100, random_state=42,class_weight='balanced')
# model = GradientBoostingClassifier(random_state=42)
# model = DecisionTreeClassifier(criterion='entropy')
# model = SVC(kernel='linear', class_weight='balanced', random_state=42)
# model = KNeighborsClassifier(n_neighbors=5)
# model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
# model = GaussianNB()

#rf_model = grid_search.best_estimator_

# Train the model
model.fit(X_train, y_train)

# COMMAND ----------

import pickle
with open('adaboost-model.pkl', 'wb') as file:
    pickle.dump(model, file)
# /Workspace/Users/jake@mujahedtrainergmail.onmicrosoft.com/Deepika/model.pkl

# COMMAND ----------

!pip install azure.storage.blob

# COMMAND ----------


from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

# Set up your connection string (found in your Azure portal under Storage Account Access keys)
connect_str = "DefaultEndpointsProtocol=https;AccountName=your_account_name;AccountKey=your_account_key;EndpointSuffix=core.windows.net"

blob_name = "adaboost-model.pkl"
path = 'adaboost-model.pkl'
blob_service_client = BlobServiceClient.from_connection_string('DefaultEndpointsProtocol=https;AccountName=deepikastorageacc;AccountKey=MFjQJGDyj4MPqU/9L+B5oIb56r+LsRlC65jD7SfjXi36go5WoCQbs//JFj/Jrbixnww+D9hyMcu0+AStSUUn4A==;EndpointSuffix=core.windows.net')
blob_client = blob_service_client.get_blob_client(container='deepika-container', blob=blob_name)
with open(path, "rb") as data:
    blob_client.upload_blob(data, overwrite=True)




# COMMAND ----------

# MAGIC %md
# MAGIC **VALIDATE THE MODEL AND GET METRICS**

# COMMAND ----------

# Step 4: Evaluate the Model
# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Display the classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# COMMAND ----------



# COMMAND ----------

