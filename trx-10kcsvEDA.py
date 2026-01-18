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

# Identify numeric and non-numeric columns
numeric_cols = [
    c for c, t in df.dtypes
    if t in ["int", "bigint", "double", "float"]
]

non_numeric_cols = [
    c for c in df.columns if c not in numeric_cols
]

# Standardize categorical columns
for c in non_numeric_cols:
    df = df.withColumn(
        c,
        lower(
            trim(
                regexp_replace(col(c), "\\s+", " ")
            )
        )
    )

# Unify status values
df = df.withColumn(
    "status",
    when(col("status").isin(
        "success", "succeed", "successful", "succeded", "ok", "completed"
    ), "Success")
    .when(col("status").isin(
        "fail", "failed", "failure", "unsuccessful", "declined", "error"
    ), "Fail")
    .otherwise("Unknown")
)

# Unify city values
df = df.withColumn(
    "city",
    when(col("city").isin(
        "tehran", "thr", "tehr@n", "teheran", "thran", "teh ran", "t e h r a n"
    ), "Tehran")
    .otherwise(initcap(col("city")))
)

# Unify card types
df = df.withColumn(
    "card_type",
    when(col("card_type").isin(
        "visa", "vsa", "vi sa", "visa-card", "visa card"
    ), "Visa")
    .when(col("card_type").isin(
        "mastercard", "master card", "master-card", "mastcard", "mstrcrd", "mc"
    ), "MasterCard")
    .when(col("card_type").isin(
        "amex", "american express", "american-express", "am ex"
    ), "Amex")
    .otherwise(initcap(col("card_type")))
)

# Remove commas
for c in df.columns:
    df = df.withColumn(c, regexp_replace(col(c), ",", ""))

# Cast numeric columns
for c in numeric_cols:
    df = df.withColumn(c, col(c).cast(DoubleType()))

# Drop nulls and duplicates
df = df.dropna(subset=numeric_cols)
df = df.dropDuplicates()

# Feature engineering
df = df.withColumn("hour", hour(col("time")))
df = df.withColumn("date", to_date(col("time")))

# 4-hour bins
df = df.withColumn(
    "hour_bin",
    concat(
        lpad((col("hour") / 4).cast("int") * 4, 2, "0"),
        lit("-"),
        lpad(((col("hour") / 4).cast("int") * 4 + 4), 2, "0")
    )
)

# Convert to Pandas
pdf = df.select(
    "amount",
    "city",
    "card_type",
    "status",
    "hour_bin",
    "date"
).toPandas()

# Convert date format to DD/MM/YY
pdf["date"] = pd.to_datetime(pdf["date"]).dt.strftime("%d/%m/%y")

# Transaction Amount Distribution with mean
plt.figure()
sns.histplot(pdf["amount"], bins=50)
plt.axvline(pdf["amount"].mean(), linestyle="--")
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()

# Number of transactions vs time of day with mean
plt.figure()
txn_by_hour = pdf["hour_bin"].value_counts().sort_index()
txn_by_hour.plot(kind="bar")
plt.axhline(txn_by_hour.mean(), linestyle="--")
plt.title("Number Of Transactions Vs Time Of Day")
plt.xlabel("Time Of Day")
plt.ylabel("Number Of Transactions")
plt.show()

# Total transaction amount by city with mean
plt.figure()
city_amount = pdf.groupby("city")["amount"].sum().sort_values(ascending=False)
city_amount.plot(kind="bar")
plt.axhline(city_amount.mean(), linestyle="--")
plt.title("Total Transaction Amount By City")
plt.xlabel("City")
plt.ylabel("Total Amount")
plt.show()

# Transaction frequency over days with rolling mean
plt.figure()
daily_txn = pdf.groupby("date").size()
daily_txn.plot(kind="line", label="Daily Transactions")
daily_txn.rolling(window=5).mean().plot(label="Rolling Mean")
plt.title("Transaction Frequency Over Days")
plt.xlabel("Date (DD/MM/YY)")
plt.ylabel("Number Of Transactions")
plt.legend()
plt.show()

# Final schema
df.printSchema()
