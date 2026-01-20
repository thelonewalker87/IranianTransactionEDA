from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter

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

# Identify numeric and non-numeric columns
numeric_cols = [
    c for c, t in df.dtypes
    if t in ["int", "bigint", "double", "float"]
]

non_numeric_cols = [
    c for c in df.columns if c not in numeric_cols
]

# Normalize text columns
for c in non_numeric_cols:
    df = df.withColumn(
        c,
        lower(trim(regexp_replace(col(c), "\\s+", " ")))
    )

# Unify status column
df = df.withColumn(
    "status",
    when(col("status").rlike("success|succeed|completed|ok|approved"), "Success")
    .when(col("status").rlike("fail|failed|declined|error|rejected"), "Fail")
    .otherwise("Unknown")
)

# Unify city column
df = df.withColumn(
    "city",
    when(col("city").rlike("tehran|thr|tehr@n|thran|teheran|teh-ran"), "Tehran")
    .otherwise(initcap(col("city")))
)

# Unify card types
df = df.withColumn(
    "card_type",
    when(col("card_type").rlike("visa|vsa"), "Visa")
    .when(col("card_type").rlike("mastercard|master card|master-card|mastcard"), "MasterCard")
    .otherwise(initcap(col("card_type")))
)

# Remove commas from all columns
for c in df.columns:
    df = df.withColumn(c, regexp_replace(col(c), ",", ""))

# Cast numeric columns
for c in numeric_cols:
    df = df.withColumn(c, col(c).cast(DoubleType()))

# Drop rows with null numeric values
df = df.dropna(subset=numeric_cols)

# Remove duplicate rows
df = df.dropDuplicates()

# Feature engineering
df = df.withColumn("Amount_Million_IRR", col("amount") / 1_000_000)
df = df.withColumn("Log_Amount_Million_IRR", log1p(col("Amount_Million_IRR")))
df = df.withColumn("Hour", hour(col("time")))

df = df.withColumn(
    "Hour_Bin",
    concat(
        lpad((floor(col("Hour") / 4) * 4).cast("string"), 2, "0"),
        lit(":00-"),
        lpad((floor(col("Hour") / 4) * 4 + 4).cast("string"), 2, "0"),
        lit(":00")
    )
)

# Outlier threshold
quantiles = df.approxQuantile("amount", [0.25, 0.75], 0.01)
Q1, Q3 = quantiles

df = df.withColumn(
    "High_Value_Txn",
    when(col("amount") > Q3, 1).otherwise(0)
)

# Convert to Pandas
pdf = df.select(
    "Amount_Million_IRR",
    "Log_Amount_Million_IRR",
    "city",
    "card_type",
    "status",
    "Hour_Bin",
    "time"
).toPandas()

# Date formatting
pdf["Date"] = pd.to_datetime(pdf["time"]).dt.strftime("%d/%m/%y")

# Formatter for readability
million_formatter = FuncFormatter(lambda x, _: f"{x:,.1f}")

# Transaction Amount Distribution (Log-scaled)
plt.figure()
sns.histplot(pdf["Log_Amount_Million_IRR"], bins=50, stat="density")
sns.kdeplot(pdf["Log_Amount_Million_IRR"])

plt.axvline(pdf["Log_Amount_Million_IRR"].mean(), linestyle="--")

plt.title("Transaction Amount Distribution (Million IRR, Log Scaled)")
plt.xlabel("Transaction Amount (Million IRR)")
plt.ylabel("Density")
plt.show()

# City vs Total Transaction Amount
city_amount = pdf.groupby("city")["Amount_Million_IRR"].sum().sort_values(ascending=False)

plt.figure()
city_amount.plot(kind="bar")
plt.axhline(city_amount.mean(), linestyle="--")

plt.title("Total Transaction Amount By City")
plt.xlabel("City")
plt.ylabel("Total Amount (Million IRR)")
plt.gca().yaxis.set_major_formatter(million_formatter)
plt.show()

# Transaction probability by time of day
txn_by_hour = pdf["Hour_Bin"].value_counts().sort_index()
txn_prob = txn_by_hour / txn_by_hour.sum()

plt.figure()
sns.lineplot(x=txn_prob.index, y=txn_prob.values, marker="o")

plt.title("Transaction Probability By Time Of Day")
plt.xlabel("Time Of Day")
plt.ylabel("Transaction Probability")
plt.xticks(rotation=45)
plt.show()

# Daily transaction frequency wave plot
daily_txn = pdf.groupby("Date").size()

plt.figure()
sns.lineplot(x=daily_txn.index, y=daily_txn.values)
sns.lineplot(
    x=daily_txn.index,
    y=daily_txn.rolling(window=7, min_periods=1).mean(),
    linestyle="--"
)

plt.title("Daily Transaction Frequency Over Time")
plt.xlabel("Date (DD/MM/YY)")
plt.ylabel("Number Of Transactions")
plt.xticks(rotation=45)
plt.show()

# Final schema
df.printSchema()
