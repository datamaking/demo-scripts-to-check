from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType
from sentence_transformers import SentenceTransformer
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder.appName("SentenceEmbeddings").getOrCreate()

# Sample DataFrame
data = [{"text": "This is a test sentence"}, {"text": "Another sentence"}]
df = spark.createDataFrame(data)

# Define the Pandas UDF
@pandas_udf(ArrayType(FloatType()))
def encode_udf(text_series: pd.Series) -> pd.Series:
    print(len(text_series) + 10)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_series.tolist())
    return pd.Series(embeddings.tolist())

# Apply the UDF
df_with_embeddings = df.withColumn("embeddings", encode_udf(df["text"]))

# Show the result
df_with_embeddings.show(truncate=False)

# Stop the Spark session
spark.stop()