from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType
from sentence_transformers import SentenceTransformer
import pandas as pd
import time

start = time.time()

# Initialize Spark session
spark = SparkSession.builder.appName("SentenceEmbeddings").getOrCreate()

# Sample DataFrame
data = [{"text": "This is a test sentence - 11"}, {"text": "Another sentence - 22"},
        {"text": "This is a test sentence - 33"}, {"text": "Another sentence - 44"},
        {"text": "This is a test sentence - 55"}, {"text": "Another sentence - 66"}]

data = [{"text": f"This is a test sentence - {i}"} for i in range(1, 10001)]
print("len(data) ==================== >")
print(len(data))

df = spark.createDataFrame(data)

# Define the Pandas UDF
@pandas_udf(ArrayType(FloatType()))
def encode_udf(text_series: pd.Series) -> pd.Series:
    #print(text_series)
    print(len(text_series))
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('sentence-t5-base')
    embeddings = model.encode(text_series.tolist())
    return pd.Series(embeddings.tolist())

# Apply the UDF
df_with_embeddings = df.withColumn("embeddings", encode_udf(df["text"]))

# Show the result
df_with_embeddings.show(50, truncate=False)
print("df_with_embeddings.count() ===========>")
print(df_with_embeddings.count())

duration = time.time() - start
print(f"Duration: {duration:.2f} seconds")

# Stop the Spark session
spark.stop()