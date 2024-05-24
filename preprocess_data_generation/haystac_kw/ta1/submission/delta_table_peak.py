import pyspark
from delta import configure_spark_with_delta_pip
import time
from pyspark.sql.functions import from_utc_timestamp
import os


if __name__ == '__main__':
    start_time = time.time()
    builder = pyspark.sql.SparkSession.builder.appName("MyApp3") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")\
        .config('spark.driver.memory', '42g')\
        .config('spark.driver.cores', '4')\
        .config('spark.jars.packages', 'io.delta:delta-core_2.12:2.3.0')\
        .config('spark.sql.parquet.compression.codec', 'zstd')\
        .config('spark.sql.parquet.compression.codec.zstd.level', 3)\
        .config('spark.sql.parquet.outputTimestampType', 'TIMESTAMP_MILLIS')\
        .config('spark.sql.session.timeZone', '+00:00')\
        .config('spark.hadoop.parquet.writer.version', 'v2')

    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    stime = time.time()
    # read in multiple parquets this works if the agent id is in the data
    #df = spark.read.format("delta").load("./humonet_train_v2.1.1")
    #df = spark.read.format("delta").load("./humonet_test_v2.1.1")
    #df = spark.read.format("delta").load("../haystac_s3_te_example/ta1.baseline/ta1/simulation/train")
    #df = spark.read.format("delta").load("./train_20230630")
    df = spark.read.format("delta").load("./agents")

    # apply a time transform
    #df = df.withColumn("timestamp", from_utc_timestamp(df["timestamp"], "+8"))

    df.show()

    b = df.tail(1)
    spark.createDataFrame(b).show()

    # specify the timezone so we don't get it converted to local
    os.environ['TZ'] = 'Europe/London'
    time.tzset()

    print(f"last item:\n{b[0]}\n")

    a = df.first().asDict()
    b = df.tail(1)[-1].asDict()
    print(f"first item: {a}\n")
    print(f"last: {b}\n")

    print(f"dtypes: \n{df.dtypes}\n")

    print(f">>>>>>> data grab time : {time.time() - stime}")  # 51s
