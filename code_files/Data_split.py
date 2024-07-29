# Databricks notebook source
import pandas as pd
from sklearn.model_selection import train_test_split

# COMMAND ----------

from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("example").getOrCreate()

# Access the global temporary views
transformed_data_spark = spark.sql("SELECT * FROM global_temp.table2")

# Show the data
transformed_data_spark.show()

# convert to pd
data_encoded = transformed_data_spark.toPandas()

# COMMAND ----------

# Separate features (X) and target (y)
X = data_encoded.drop('Default', axis=1)
y = data_encoded['Default']

# Split the balanced data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Combine the features and target back into DataFrames for easier handling
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# Display the distribution of the target variable in the train and test sets
print("Training set target distribution:\n", train['Default'].value_counts(normalize=True))
print("Test set target distribution:\n", test['Default'].value_counts(normalize=True))

# COMMAND ----------

len(y_test)

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, IntegerType
# Step 2: Convert Pandas DataFrames to Spark DataFrames
X_train_spark = spark.createDataFrame(X_train)
X_test_spark = spark.createDataFrame(X_test)

# Convert Series to DataFrame
y_train_df = y_train.to_frame(name='target')
y_test_df = y_test.to_frame(name='target')

# Create Spark DataFrame with explicit schema
from pyspark.sql.types import IntegerType, StructType, StructField
schema = StructType([StructField("target", IntegerType(), True)])

y_train_spark = spark.createDataFrame(y_train_df, schema)
y_test_spark = spark.createDataFrame(y_test_df, schema)

# COMMAND ----------

# Step 3: Create Global Temporary Views
X_train_spark.createOrReplaceGlobalTempView("table_X_train_spark")
X_test_spark.createOrReplaceGlobalTempView("table_X_test_spark")
y_train_spark.createOrReplaceGlobalTempView("table_y_train_spark")
y_test_spark.createOrReplaceGlobalTempView("table_y_test_spark")


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

