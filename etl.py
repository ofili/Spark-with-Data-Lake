import configparser
import os

from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, from_unixtime, to_date
from pyspark.sql.functions import year, month, dayofmonth, dayofweek, hour, weekofyear, date_format
from pyspark.sql.types import StructType, StructField as Sd, DoubleType as Dbl, StringType as Str, IntegerType as Int, \
    DateType as Date, TimestampType as Timestamp

config = configparser.ConfigParser()
config.read('dl.cfg')

AWS_ACCESS_KEY_ID = config.get('AWS','AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = config.get('AWS','AWS_SECRET_ACCESS_KEY')


def create_spark_session():
    """
    Create a Spark session

    Returns:
        SparkSession: Spark session
    """
    AWS_ACCESS_KEY_ID = config['AWS']['AWS_ACCESS_KEY_ID']
    AWS_SECRET_ACCESS_KEY = config['AWS']['AWS_SECRET_ACCESS_KEY']

    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()

    spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY)
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY_ID)
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.amazonaws.com")
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3n.endpoint", "s3.amazonaws.com")

    return spark


def song_schema():
    """
    Create a schema for the songs table

    Returns:
        StructType: Schema for the songs table
    """
    return StructType([
        Sd("num_songs", Int()),
        Sd("song_id", Str()),
        Sd("title", Str()),
        Sd("artist_id", Str()),
        Sd("artist_name", Str()),
        Sd("artist_latitude", Dbl()),
        Sd("artist_longitude", Dbl()),
        Sd("artist_location", Str()),
        Sd("year", Int()),
        Sd("duration", Dbl())
    ])


''' def create_artists_table():
    """
    Create a schema for the artists table

    Returns:
        StructType: Schema for the artists table
    """
    return StructType([
        Sd("artist_id", Str()),
        Sd("artist_name", Str()),
        Sd("artist_location", Str()),
        Sd("artist_latitude", Dbl()),
        Sd("artist_longitude", Dbl())
    ])
'''

def process_song_data(spark, input_data, output_data):
    """
    Process song data

    Args:
        spark: Spark session
        input_data: Input data location
        output_data: Output data location
    Return:
        None
    """
    # get filepath to song data file
    song_data = input_data + "song_data/*/*/*/*.json"

    # read song data file
    df = spark.read.json(song_data, schema=song_schema(), multiLine=True)

    # extract columns to create songs table
    songs_table = df.select("song_id", "title", "artist_id", "year", "duration").dropDuplicates().where(
        col(Str("song_id").cast("string")).isNotNull())

    # write songs table to parquet files partitioned by year and artist
    song_path = output_data + "songs"
    songs_table.write.partitionBy("year", "artist_id").parquet(song_path, mode="overwrite")

    # extract columns to create artists table
    artists_table = df.select("artist_id", "artist_name", "artist_location", "artist_latitude",
                            "artist_longitude").dropDuplicates().where(
        col(Str("artist_id").cast("string")).isNotNull())

    # write artists table to parquet files
    artist_path = output_data + "artists"
    artists_table.write.parquet(artist_path, mode="overwrite")


def process_log_data(spark, input_data, output_data):
    """
    Process log data

    Args:
        spark: Spark session
        input_data: Input data location
        output_data: Output data location
    """
    # get filepath to log data file
    log_data = input_data + "log_data/*/*/*.json"

    # read log data file
    df = spark.read.json(log_data, multiLine=True)

    # filter by actions for song plays
    df = df.where(col("page") == "NextSong")

    # extract columns for users table
    users_table = df.select("userId", "firstName", "lastName", "gender", "level").dropDuplicates().where(
        col("userId").cast("string").isNotNull())

    # write users table to parquet files
    user_path = output_data + "users"
    users_table.write.parquet(user_path, mode="overwrite")


def format_datetime(ts):
    """
    Format datetime from numeric

    Args:
        ts: Timestamp

    Returns:
        str: Formatted datetime
    """
    return datetime.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: format_datetime(int(x)), Timestamp())
    df = df.withColumn("start_time", get_timestamp(col("ts")))

    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: format_datetime(int(x)), Date())
    df = df.withColumn("start_time", get_datetime(col("ts")))

    # extract columns to create timetable
    time_table = df.select("ts", "start_time", "datetime", hour(col("start_time")).alias("hour"),
                        dayofmonth(col("start_time")).alias("day"), weekofyear(col("start_time")).alias("week"),
                        month(col("start_time")).alias("month"), year(col("start_time")).alias("year"),
                        dayofweek(col("start_time")).alias("weekday")
                        ).dropDuplicates().withColumn("start_time", col("start_time").cast("timestamp"))

    # write timetable to parquet files partitioned by year and month
    time_path = output_data + "time"
    time_table.write.partitionBy("year", "month").parquet(time_path, mode="overwrite")

    # read in song data to use for songplays table
    song_data = input_data + "song_data/*/*/*/*.json"
    song_df = spark.read.json(song_data, schema=song_schema(), multiLine=True)

    # extract columns from joined song and log datasets to create songplays table
    songplays_table = df.join(song_df, (df.song == song_df.title) & (df.artist == song_df.artist_name) & (
            df.length == song_df.duration), "left").select(
        col("start_time").cast("timestamp").alias("start_time"), col("userId").alias("user_id"), col("level"),
        col("song_id"), col("artist_id"), col("sessionId").alias("session_id"), col("location"),
        col("userAgent").alias("user_agent"), col("year"), col("month")
    ).withColumn("songplay_id", monotonically_increasing_id()).withColumn("start_time",
                                                                        col("start_time").cast("timestamp"))

    # write songplays table to parquet files partitioned by year and month
    songplay_path = output_data + "songplays"
    songplays_table.write.partitionBy("year", "month").parquet(songplay_path, mode="overwrite")


def main():
    """
    Description: Calls functions to create spark session, read from S3 and perform ETL to S3 Data Lake.
    """
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://udacity-dend-output/"

    print("Processing song data")
    process_song_data(spark, input_data, output_data)
    print("Processing song data complete")
    print("Processing log data")
    process_log_data(spark, input_data, output_data)
    print("Processing log data complete")

if __name__ == "__main__":
    main()