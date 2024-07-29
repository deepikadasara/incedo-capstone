# Databricks notebook source
import pandas as pd
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# COMMAND ----------

# get the data from previous notebook
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("example").getOrCreate()

# Access the global temporary views
data_spark = spark.sql("SELECT * FROM global_temp.table1")

# Show the data
data_spark.show()

#convert to pandas
data_pd = data_spark.toPandas()


# COMMAND ----------

# MAGIC %md
# MAGIC **ENCODE THE CATEGORICAL COLUMNS**

# COMMAND ----------

categorical_columns = ['Education',
 'EmploymentType',
 'MaritalStatus',
 'HasMortgage',
 'HasDependents',
 'LoanPurpose',
 'HasCoSigner']

 

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Identify categorical columns
#categorical_columns = data_pd.select_dtypes(include=['object']).columns.tolist()

# Apply one-hot encoding to categorical columns using sparse matrix
encoder = OneHotEncoder(drop='first', sparse=True)
#encoder = OneHotEncoder(drop='first', sparse=False)
encoded_categorical = encoder.fit_transform(data_pd[categorical_columns])

# Convert the sparse matrix to a sparse DataFrame
encoded_categorical_df = pd.DataFrame.sparse.from_spmatrix(
    encoded_categorical,
    columns=encoder.get_feature_names_out(categorical_columns)
)

# Combine encoded categorical columns with numeric columns
numeric_columns = data_pd.select_dtypes(include=['number']).columns.tolist()
data_encoded = pd.concat([data_pd[numeric_columns].reset_index(drop=True), encoded_categorical_df], axis=1)

# Display the encoded DataFrame
print(data_encoded.head())

# COMMAND ----------

data_encoded.info()

# COMMAND ----------

# Step 2: Convert Pandas DataFrames to Spark DataFrames
data_encoded_spark = spark.createDataFrame(data_encoded)

# Step 3: Create Global Temporary Views
data_encoded_spark.createOrReplaceGlobalTempView("table2")

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



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

