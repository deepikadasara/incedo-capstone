# Databricks notebook source
# MAGIC %md
# MAGIC Load Libraries
# MAGIC

# COMMAND ----------

import pandas as pd
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# COMMAND ----------

# MAGIC %md
# MAGIC Create connection with Loan data stored in storage container and load it

# COMMAND ----------

# Define the storage account details
storage_account_name = 'deepikastorageacc'
storage_account_access_key = 'MFjQJGDyj4MPqU/9L+B5oIb56r+LsRlC65jD7SfjXi36go5WoCQbs//JFj/Jrbixnww+D9hyMcu0+AStSUUn4A=='
container_name = 'deepika-container'
mount_name = "/mnt/deepikastorageacc/deepika-container"

# Mount the storage account
dbutils.fs.mount(
  source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net",
  mount_point = mount_name,
  extra_configs = {f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_access_key}
)

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
# MAGIC Data Exploration

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

# COMMAND ----------

categorical_columns

# COMMAND ----------

numeric_columns

# COMMAND ----------

# finding out how balanced is the data
data_pd['Default'].value_counts()

# COMMAND ----------

# check for nulls
data_pd.isnull().sum()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

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

# # Plot each categorical feature against the Default column using count plots
# categorical_columns = data_pd.select_dtypes(include=['object']).columns.tolist()

# plt.figure(figsize=(20, 20))
# for i, col in enumerate(categorical_columns):
#     plt.subplot(5, 2, i + 1)
#     sns.countplot(data=data_pd, x=col, hue='Default')
#     plt.title(f'{col} vs Default')
#     plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()


# COMMAND ----------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Assuming data_pd is your DataFrame

# # Get categorical columns
# categorical_columns = data_pd.select_dtypes(include=['object']).columns.tolist()

# # Split the categorical columns into two batches
# batch1 = categorical_columns[:len(categorical_columns)//2]
# batch2 = categorical_columns[len(categorical_columns)//2:]

# # Function to plot a batch of features
# def plot_categorical_batch(batch, data, title):
#     plt.figure(figsize=(20, 10))
#     for i, col in enumerate(batch):
#         plt.subplot(2, len(batch)//2, i + 1)
#         sns.countplot(data=data, x=col, hue='Default')
#         plt.title(f'{col} vs Default')
#         plt.xticks(rotation=90)
#     plt.tight_layout()
#     plt.suptitle(title, y=1.02)
#     plt.show()

# # Plot the first batch
# plot_categorical_batch(batch1, data_pd, "Batch 1: Categorical Features vs Default")

# # Plot the second batch
# plot_categorical_batch(batch2, data_pd, "Batch 2: Categorical Features vs Default")


# COMMAND ----------

# MAGIC %md
# MAGIC Balance the data

# COMMAND ----------

import pandas as pd

# Assuming data_pd is your DataFrame
# Separate the Default and No Default rows
default_df = data_pd[data_pd['Default'] == 1]
no_default_df = data_pd[data_pd['Default'] == 0]

print("Number of Default rows:", len(default_df))
print("Number of No Default rows:", len(no_default_df))

# Balance the number of Default and No Default rows
num_default = len(default_df)
# balanced_no_default_df = no_default_df.sample(n=num_default, random_state=42)
#num_default = 5000
balanced_no_default_df = no_default_df.sample(n=num_default, random_state=42)

# Combine the two DataFrames to create a balanced DataFrame
balanced_df = pd.concat([default_df, balanced_no_default_df])

# Shuffle the balanced DataFrame
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Balanced DataFrame shape:", balanced_df.shape)
print("Balanced DataFrame Default value counts:\n", balanced_df['Default'].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC Encode the categorical columns
# MAGIC

# COMMAND ----------

data_pd['LoanID'].value_counts()

# COMMAND ----------

categorical_columns = balanced_df.select_dtypes(include=['object']).columns.tolist()
categorical_columns = ['Education',
 'EmploymentType',
 'MaritalStatus',
 'HasMortgage',
 'HasDependents',
 'LoanPurpose',
 'HasCoSigner']

# COMMAND ----------

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder

# # Identify categorical columns
# #categorical_columns = balanced_df.select_dtypes(include=['object']).columns.tolist()

# # Apply one-hot encoding to categorical columns using sparse matrix
# encoder = OneHotEncoder(drop='first', sparse=True)
# #encoder = OneHotEncoder(drop='first', sparse=False)
# encoded_categorical = encoder.fit_transform(balanced_df[categorical_columns])

# # Convert the sparse matrix to a sparse DataFrame
# encoded_categorical_df = pd.DataFrame.sparse.from_spmatrix(
#     encoded_categorical,
#     columns=encoder.get_feature_names_out(categorical_columns)
# )

# # Combine encoded categorical columns with numeric columns
# numeric_columns = balanced_df.select_dtypes(include=['number']).columns.tolist()
# data_encoded = pd.concat([balanced_df[numeric_columns].reset_index(drop=True), encoded_categorical_df], axis=1)

# # Display the encoded DataFrame
# print(data_encoded.head())

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

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Split the data

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

# drop loanid column
#data_encoded = data_encoded.drop(columns=['LoanID'])

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

len(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Train the model

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


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
model = RandomForestClassifier(n_estimators=100, random_state=42)
#model = GradientBoostingClassifier(random_state=42)
#model = DecisionTreeClassifier(criterion='entropy')
#rf_model = grid_search.best_estimator_

# Train the model
model.fit(X_train, y_train)


# COMMAND ----------

# MAGIC %md
# MAGIC Test

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