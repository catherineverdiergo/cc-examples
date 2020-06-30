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
from pyspark.sql.types import StructType, StructField, StringType, DateType, FloatType, ArrayType

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


def download_wet_files(spark, cc_crawl_id, num_files=10, start_at=0):
    wet_paths_file = download_https_cc_file("crawl-data/" + cc_crawl_id + "/wet.paths.gz")
    list_files = spark.sparkContext.textFile(wet_paths_file).take(num_files)
    result = []
    if start_at > 0:
        list_files = list_files[start_at:]
    for wet_path in list_files:
        local_wet_path = download_https_cc_file(wet_path)
        print(local_wet_path)
        result.append(local_wet_path)
    return result


def get_country(reader, domain):
    try:
        country = reader.get(socket.gethostbyname(domain))
        if country:
            if country.get('location') and country.get('location').get('longitude') and \
                    country.get('location').get('latitude'):
                loc = [country.get('location').get('longitude'), country.get('location').get('latitude')]
            else:
                loc = None
            country = country.get('country').get('iso_code')
            country = [country, loc]
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
        StructField("lang", StringType()),
        StructField("country", StringType()),
        StructField("loc", ArrayType(FloatType())),
        StructField("text", StringType())
    ])
    if with_country:
        cSchema.add(StructField("country", StringType()))
    list_rows = []
    reader = geolite2.reader()
    for record in f:
        url = record.header.get('warc-target-uri', None)
        if not url:
            continue
        text = record.payload.read().decode('utf8')
        lang = langid.classify(text)[0]
        if lang not in ['en', 'fr']:
            continue
        domain = urlparse(url).netloc
        location = get_country(reader, domain)
        if location is None:
            country = None
            loc = None
        else:
            country = location[0]
            loc = location[1]
        crawl_date = datetime.strptime(record.header.get('warc-date', None), "%Y-%m-%dT%H:%M:%SZ").date()
        row = [url, domain, crawl_date, lang, country, loc, text]
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
        print("File {} has been processed".format(file))
    return df


if __name__ == '__main__':
    spark = SparkSession.builder.appName('wetstatsoncorpus').getOrCreate()
    list_files = download_wet_files(spark, sys.argv[1], num_files=4, start_at=0)
    with_country = False
    df = load_corpus(spark, list_files, with_country=with_country)
    df.show()
    (df.repartition("lang")
       .repartition(5)
       .write.mode("overwrite")
       .partitionBy("lang")
       .parquet(sys.argv[2]))
    spark.stop()
