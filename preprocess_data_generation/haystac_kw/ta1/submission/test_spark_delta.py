import os
from glob import glob
import pyspark
from delta import configure_spark_with_delta_pip
import time
from pyspark.sql.functions import lit
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import format_number

# note: pyspark==3.3.2
# delta-spark==2.3.0
memory = '42g'
cores = '10'


if __name__ == '__main__':
    global spark
    start_time = time.time()
    builder = pyspark.sql.SparkSession.builder.appName("MyApp3") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")\
        .config('spark.driver.memory', memory)\
        .config('spark.driver.cores', cores)\
        .config('spark.jars.packages', 'io.delta:delta-core_2.12:2.3.0')\
        .config('spark.sql.parquet.compression.codec', 'zstd')\
        .config('spark.sql.parquet.compression.codec.zstd.level', 3)\
        .config('spark.sql.parquet.outputTimestampType', 'TIMESTAMP_MILLIS')\
        .config('spark.sql.session.timeZone', '+00:00')\
        .config('spark.hadoop.parquet.writer.version', 'v2')

    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    # specify the timezone so we don't get it converted to local
    os.environ['TZ'] = 'Europe/London'
    time.tzset()

    # test with 2 files
    loc1 = "./humonet_train_v2.1.1/agent=0a0b34b6-0fc6-46a7-a7db-7fa0050cf040/part-00290-3681b8aa-2655-414e-bedf-52de49d93e13.c000.zstd.parquet"
    loc2 = "./humonet_train_v2.1.1/agent=0a0bebbe-c7d6-4761-8ff9-1dc297d5d0f2/part-00035-2a1ec69b-205f-475e-89c4-8c455983d0ec.c000.zstd.parquet"
    # files = [loc1, loc2]  # 60s using the pd.concat method

    # read in all the parquet files
    files = glob('./translated_files/*.parquet')

    stime = time.time()
    # read in multiple parquets this works if the agent id is in the data
    df = spark.read.parquet(*files)
    # reorder to match spec.
    df1 = df.selectExpr("agent",
                        "timestamp",
                        "longitude",  # "Longitude as longitude",  # looks like we don't need this anymore
                        "latitude"  # "Latitude as latitude"
                        )
    df1.show()
    print(f">>>>>>> grabbed all parquet files: {time.time() - stime}")  # 51s

    # coalesce to a single parquet file and sort by timestamp
    stime = time.time()
    df2 = df1 \
        .coalesce(1)\
        .sort("agent", "timestamp")
    print(f">>>>>>> coalesce and sort: {time.time() - stime}")  # 0.07s

    # from here:
    # https://haystac-program.com/testing-and-evaluation/interfaces-and-standards/-/wikis/Standards/Conventions#delta
    options = {
        'spark.jars.packages': 'io.delta:delta-core_2.12:2.3.0',
        'spark.sql.catalog.spark_catalog': 'org.apache.spark.sql.delta.catalog.DeltaCatalog',
        'spark.sql.extensions': 'io.delta.sql.DeltaSparkSessionExtension',
        'spark.sql.parquet.compression.codec': 'zstd',
        'spark.sql.parquet.compression.codec.zstd.level': 3,
        'spark.sql.parquet.outputTimestampType': 'TIMESTAMP_MILLIS',
        'spark.sql.session.timeZone': '+00:00',
        'spark.hadoop.parquet.writer.version': 'v2'
    }

    # truncate the numbers
    df2 = df2.withColumn("longitude",
                         format_number(df.longitude, 6))
    df2 = df2.withColumn("latitude",
                         format_number(df.latitude, 6))

    # convert the column types back to doubles
    df2 = df2.withColumn("longitude", df2["longitude"].cast(DoubleType()))
    df2 = df2.withColumn("latitude", df2["latitude"].cast(DoubleType()))

    spark_dataframe_sql = df2.withColumn("agent",
                                         df2["agent"].cast(IntegerType()))
    print(f"dtypes: \n{spark_dataframe_sql.dtypes}\n")
    print(f"schema: \n{spark_dataframe_sql.schema}\n")

    # get all the current settings in spark
    # print(spark.sparkContext.getConf().getAll())

    # review the items
    a = spark_dataframe_sql.first().asDict()
    b = spark_dataframe_sql.tail(1)[-1].asDict()
    print(f"first item: {a}\n")
    print(f"last: {b}\n")

    stime = time.time()
    spark_dataframe_sql.write.partitionBy("agent")\
        .mode("overwrite")\
        .format("delta")\
        .option("overwriteSchema", "true")\
        .option('outputTimestampType', 'TIMESTAMP_MILLIS')\
        .option('timeZone', '+00:00')\
        .option('parquet.writer.version', 'v2')\
        .option('compression', 'zstd')\
        .option('level', 3)\
        .option("maxRecordsPerFile", "-1")\
        .save("./train")
    print(f">>>>>>> save time: {time.time() - stime}")

    # create the agents table
    stime = time.time()
    agent_rows = spark_dataframe_sql.select('agent').distinct().collect()
    agents = [a.asDict()['agent'] for a in agent_rows]
    adf = spark.createDataFrame(agents, IntegerType()).sort("value")
    agent_table = adf.selectExpr("value as agent_uid")
    agent_table.show()
    agent_table.write.mode("overwrite")\
        .format("delta")\
        .option('parquet.writer.version', 'v2')\
        .option('compression', 'zstd')\
        .option('level', 3)\
        .option("maxRecordsPerFile", "-1")\
        .save("./agents")
    print(f">>>>>>> agent save time: {time.time() - stime}")

    spark.stop()

    print(f"final time: {time.time() - start_time}")
