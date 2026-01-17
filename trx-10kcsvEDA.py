from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Spark Session
spark = SparkSession.builder \
    .appName("Iran Transaction Full EDA") \
    .getOrCreate()

# Load Dataset
df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .option("mode", "DROPMALFORMED") \
    .csv("C:\\Users\\Ahaan\\Desktop\\AI\\IranTransactionEDA\\trx-10k.csv")

print("Initial Schema")
df.printSchema()
df.show(5)

# Identify Column Types
numeric_cols = [
    c for c, t in df.dtypes
    if t in ["int", "bigint", "double", "float"]
]

non_numeric_cols = [
    c for c in df.columns if c not in numeric_cols
]

# Null Counts (SAFE)
numeric_nulls = df.select([
    count(
        when(col(c).isNull() | isnan(col(c)), c)
    ).alias(c)
    for c in numeric_cols
])

non_numeric_nulls = df.select([
    count(
        when(col(c).isNull(), c)
    ).alias(c)
    for c in non_numeric_cols
])

numeric_nulls.show()
non_numeric_nulls.show()

# Remove commas from all columns
for c in df.columns:
    df = df.withColumn(c, regexp_replace(col(c), ",", ""))

# Cast numeric columns again
for c in numeric_cols:
    df = df.withColumn(c, col(c).cast(DoubleType()))

# Drop rows with null numeric values
df = df.dropna(subset=numeric_cols)

# Remove duplicate rows
df = df.dropDuplicates()

# Descriptive statistics
df.select(numeric_cols).describe().show()

# Feature Engineering
df = df.withColumn("log_amount", log1p(col("amount")))

# Outlier Detection (IQR)
quantiles = df.approxQuantile("amount", [0.25, 0.75], 0.01)
Q1, Q3 = quantiles
IQR = Q3 - Q1

df = df.withColumn(
    "amount_outlier",
    when(
        (col("amount") < Q1 - 1.5 * IQR) |
        (col("amount") > Q3 + 1.5 * IQR),
        1
    ).otherwise(0)
)

df.groupBy("amount_outlier").count().show()

# High-value transaction flag
df = df.withColumn(
    "high_value_txn",
    when(col("amount") > Q3, 1).otherwise(0)
)

# Extract hour from timestamp
df = df.withColumn(
    "hour",
    hour(col("time"))
)

# Convert to Pandas for Visualization
pdf = df.select(
    "amount",
    "log_amount",
    "city",
    "card_type",
    "status",
    "hour",
    "high_value_txn"
).sample(0.3).toPandas()

# Visualization
plt.figure()
sns.histplot(pdf["amount"], bins=50)
plt.title("Transaction Amount Distribution")
plt.show()

plt.figure()
sns.boxplot(x=pdf["amount"])
plt.title("Amount Outliers")
plt.show()

plt.figure()
sns.countplot(data=pdf, x="card_type")
plt.title("Transactions by Card Type")
plt.show()

plt.figure()
sns.scatterplot(x=pdf["hour"], y=pdf["amount"])
plt.title("Transaction Amount vs Hour")
plt.show()

# Simple Prediction Logic
pdf["predicted_risk"] = pdf["amount"].apply(
    lambda x: 1 if x > Q3 else 0
)

print(pdf[["amount", "predicted_risk"]].head())

# Final Schema
df.printSchema()
