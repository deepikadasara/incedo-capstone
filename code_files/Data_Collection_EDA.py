# Databricks notebook source
import pandas as pd
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC **LOAD THE DATA**

# COMMAND ----------

# Define the storage account details
storage_account_name = 'deepikastorageacc'
storage_account_access_key = 'MFjQJGDyj4MPqU/9L+B5oIb56r+LsRlC65jD7SfjXi36go5WoCQbs//JFj/Jrbixnww+D9hyMcu0+AStSUUn4A=='
container_name = 'deepika-container'
mount_name = "/mnt/deepikastorageacc/deepika-container"

# # Mount the storage account
# dbutils.fs.mount(
#   source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net",
#   mount_point = mount_name,
#   extra_configs = {f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_access_key}
# )

# COMMAND ----------

files = dbutils.fs.ls('/mnt/deepikastorageacc/deepika-container/')
for file in files:
    print(file)

# COMMAND ----------

data = spark.read.format('csv').option("inferSchema", "true").option("delimiter", ",").option('header', 'true').load("/mnt/deepikastorageacc/deepika-container/Loan_default.csv")
data.head()

# COMMAND ----------

# convert to pandas df
data_pd = data.toPandas()
print(data_pd.head())

# COMMAND ----------

# MAGIC %md
# MAGIC **DATA EXPLORATION**

# COMMAND ----------

data_pd.describe()

# COMMAND ----------

data_pd.info()

# COMMAND ----------

# find categorical, numerical columns
print(data_pd.dtypes)

# COMMAND ----------

categorical_columns = data_pd.select_dtypes(include=['object','category']).columns.tolist()
numeric_columns = data_pd.select_dtypes(include=['number','category']).columns.tolist()
print(categorical_columns)
print(numeric_columns)

# COMMAND ----------

# finding out how balanced is the data
data_pd['Default'].value_counts()

# COMMAND ----------

# check for nulls
data_pd.isnull().sum()

# COMMAND ----------

# Plot each numeric feature against the Default column using box plots
numeric_columns = data_pd.select_dtypes(include=['number']).columns.tolist()

plt.figure(figsize=(20, 20))
for i, col in enumerate(numeric_columns):
    plt.subplot(5, 2, i + 1)
    sns.boxplot(data=data_pd, x='Default', y=col)
    plt.title(f'{col} vs Default')
plt.tight_layout()
plt.show()


# COMMAND ----------

correlation_matrix = data_pd.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Histograms for numerical features
data_pd.hist(bins=15, figsize=(15, 10))
plt.show()

# Boxplots for numerical features
data_pd.plot(kind='box', subplots=True, layout=(4,4), figsize=(15, 10), sharex=False, sharey=False)
plt.show()


# COMMAND ----------

# Filter categorical columns
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
       'HasDependents', 'LoanPurpose', 'HasCoSigner']

# Bar charts for categorical variables
for column in categorical_cols:
    data_pd[column].value_counts().plot(kind='bar')
    plt.title(column)
    plt.show()


# COMMAND ----------

spark_data = spark.createDataFrame(data_pd)
spark_data.createOrReplaceGlobalTempView("table1")


# COMMAND ----------

