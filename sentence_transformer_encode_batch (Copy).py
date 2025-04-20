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
data = [{"text": f"This is a test sentence - {i}"} for i in range(1, 10001)]
print("len(data) ==================== >")
print(len(data))

df = spark.createDataFrame(data)

# Define the Pandas UDF
@pandas_udf(ArrayType(FloatType()))
def encode_udf(text_series: pd.Series) -> pd.Series:
    print(text_series)
    print(len(text_series))
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('sentence-t5-base')
    embeddings = model.encode(text_series.tolist())
    return pd.Series(embeddings.tolist())

# Apply the UDF
#df_with_embeddings = df.withColumn("embeddings", encode_udf(df["text"]))

batch_size = 1000

#model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('sentence-t5-base')

def _process_batch(texts):
    # Encode a batch of texts into embeddings
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True
    ).tolist()

# Define schema for the output DataFrame
result_schema = df.schema.add("embeddings", ArrayType(FloatType()))


# Process partitions in batches
def process_partition(iterator):
    for pdf in iterator:
        # Extract texts from the partition
        texts = pdf["text"].tolist()
        # Generate embeddings for the entire batch
        embeddings = _process_batch(texts)
        # Assign embeddings back to the DataFrame
        pdf["embeddings"] = embeddings
        yield pdf


#df_with_embeddings = df.mapInPandas(process_partition, schema=result_schema)

from pyspark.sql.types import StructType, StructField, ArrayType, FloatType

# 1️⃣ Copy the existing fields (so df.schema is untouched)
input_fields = df.schema.fields[:]

# 2️⃣ Append the embeddings field
output_fields = input_fields + [
    StructField("embeddings", ArrayType(FloatType()), nullable=False)
]

# 3️⃣ Create a fresh StructType
result_schema = StructType(output_fields)

# 4️⃣ Now you can safely call mapInPandas
df_with_embeddings = df.mapInPandas(process_partition, schema=result_schema)


# Show the result
df_with_embeddings.show(100, truncate=False)

print("df_with_embeddings.count() ===========>")
print(df_with_embeddings.count())

duration = time.time() - start
print(f"Duration: {duration:.2f} seconds")

# Stop the Spark session
spark.stop()