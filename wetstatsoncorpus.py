# -*- coding: utf-8 -*-

# Created by cverdier at 02/06/2020

import requests
import os
import sys
import warc
import langid
import socket
from datetime import datetime
from urllib.parse import urlparse
from geolite2 import geolite2
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType

TMP_DIR = "/tmp"

COMMON_CRAWL_HTTP_ROOT = "https://commoncrawl.s3.amazonaws.com"


def download_https_cc_file(file_name):
    """
    Download file from common-crawl database using https
    :param file_name:
    :return:
    """
    target_file = TMP_DIR + "/" + file_name
    target_dir_name = os.path.dirname(target_file)
    if not os.path.exists(target_file):
        if not os.path.exists(target_dir_name):
            os.makedirs(target_dir_name)
        url = COMMON_CRAWL_HTTP_ROOT + "/" + file_name
        r = requests.get(url, allow_redirects=True)
        open(target_file, 'wb').write(r.content)
    return target_file


def download_wet_files(spark, cc_crawl_id, num_files=10):
    wet_paths_file = download_https_cc_file("crawl-data/" + cc_crawl_id + "/wet.paths.gz")
    list_files = spark.sparkContext.textFile(wet_paths_file).take(num_files)
    result = []
    for wet_path in list_files:
        local_wet_path = download_https_cc_file(wet_path)
        print(local_wet_path)
        result.append(local_wet_path)
    return result


def get_country(reader, domain):
    try:
        country = reader.get(socket.gethostbyname(domain))
        if country:
            country = country.get('country').get('iso_code')
        else:
            country = None
    except:
        country = None
    return country


def load_wet_in_df(spark, wet_file, with_country=False):
    f = warc.open(wet_file)
    cSchema = StructType([
        StructField("url", StringType()),
        StructField("domain", StringType()),
        StructField("date", DateType()),
        StructField("lang", StringType())
    ])
    if with_country:
        cSchema.add(StructField("country", StringType()))
    list_rows = []
    reader = geolite2.reader()
    for record in f:
        url = record.header.get('warc-target-uri', None)
        if not url:
            continue
        domain = urlparse(url).netloc
        country = get_country(reader, domain)
        crawl_date = datetime.strptime(record.header.get('warc-date', None), "%Y-%m-%dT%H:%M:%SZ").date()
        text = record.payload.read()
        if len(text) > 500:
            text = text[0:500]
        lang = langid.classify(text)[0]
        row = [url, domain, crawl_date, lang]
        if with_country:
            row.append(country)
        list_rows.append(row)
    f.close()
    geolite2.close()
    return spark.createDataFrame(list_rows, schema=cSchema)


def load_corpus(spark, list_files, with_country=False):
    df = None
    for file in list_files:
        if df is None:
            df = load_wet_in_df(spark, file, with_country=with_country)
        else:
            df = df.union(load_wet_in_df(spark, file, with_country=with_country))
    return df


def stats(df, field, pct=False):
    if pct:
        all_pages = df.count()
        pct_udf = udf(lambda c: c/all_pages*100, DoubleType())
        return (df.groupBy(field)
                .count()
                .sort("count", ascending=False)
                .withColumn("pct", pct_udf(col("count"))))
    else:
        return (df.groupBy(field)
                .count()
                .sort("count", ascending=False))


def stats_2_fields(df, field1, field2):
    return (df.groupBy(field1, field2)
            .count()
            .sort("count", ascending=False))


if __name__ == '__main__':
    spark = SparkSession.builder.appName('wetstatsoncorpus').getOrCreate()
    list_files = download_wet_files(spark, sys.argv[1], num_files=4)
    with_country = False
    df = load_corpus(spark, list_files, with_country=with_country)
    df.write.mode("overwrite").parquet(sys.argv[2])
    df.cache()
    df.show()
    stats(df, "lang", pct=True).show(truncate=False)
    stats(df, "domain").show(truncate=False)
    if with_country:
        stats(df, "country", pct=True).show(truncate=False)
    df.unpersist()
    spark.stop()
