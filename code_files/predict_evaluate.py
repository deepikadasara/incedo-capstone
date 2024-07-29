# Databricks notebook source
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

import pickle
model_path = "/dbfs/mnt/deepikastorageacc/deepika-container/adaboost-model.pkl"
 
# Load the pickle file
with open(model_path, 'rb') as file:
    model = pickle.load(file)



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