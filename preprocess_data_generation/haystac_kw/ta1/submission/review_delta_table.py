import pyspark
from delta import configure_spark_with_delta_pip
from pyspark.sql.functions import countDistinct
from rich import print
import json
import time
import os
import datetime

# note: pyspark==3.3.2
# delta-spark==2.3.0
memory = '42g'
cores = '10'


if __name__ == '__main__':
    builder = pyspark.sql.SparkSession.builder.appName("MyApp") \
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
    ps = configure_spark_with_delta_pip(builder).getOrCreate()

    df = ps.read.format("delta").load("./train/")
    print(f"spark data frame: {df}")

    # specify the timezone so we don't get it converted to local
    os.environ['TZ'] = 'Europe/London'
    time.tzset()

    for x in df.take(5):
        print(x.asDict())
    for x in df.tail(5):
        print(x.asDict())

    # show duplicates (turning this off since it is checked prior)
    '''
    print("duplicates:")
    dups = df \
        .groupby(['agent', 'timestamp']) \
        .count() \
        .where('count > 1') \
        .sort('count', ascending=False)
    dups.show()
    dups_found = dups.count()
    print(f"duplicates: {dups_found}\n")
    '''

    # check for the number of agents
    dagents = df.select(countDistinct("agent"))
    print("agents = ")
    dagents.show()

    # check for the start and stop times
    a = df.first().asDict()
    b = df.tail(1)[-1].asDict()
    print(f"time start: {a['timestamp']} to {b['timestamp']}")
    iso1 = a['timestamp'].isoformat() + 'Z'
    iso2 = b['timestamp'].isoformat() + 'Z'

    # add the following times
    c = b['timestamp'] + datetime.timedelta(seconds=1)
    iso3 = c.isoformat() + 'Z'
    d = c + datetime.timedelta(seconds=(7 * 24 * 60 * 60) - 1)
    iso4 = d.isoformat() + 'Z'

    # look at the last 2 entries
    print("last 2 entries:")
    b = df.tail(2)
    print(b)

    print(f"dtypes of the columns (agent:int, timestamp:timestamp, ll:doubles): \n{df.dtypes}\n")

    # TODO see if the agents have the correct number of entries
    # 604800 seconds is supposed to be the correct amount
    res = df.groupBy("agent").count()
    print(f"count of agent values should be (604800): {res.count()}")

    # generate meta data for these
    times = {
        "schema_version": "1.1.2",
        "schema_type": "SSS.times",
        "timezone": "+08:00",
        "train": {
            "time_window": {
                "begin": iso1,
                "end": iso2
            }
        },
        "test": {
            "time_window": {
                "begin": iso3,
                "end": iso4
            }
        }
    }
    print(times)

    # save the times.json
    with open("./times.json", "w") as jf:
        json.dump(times, jf, indent=4)
