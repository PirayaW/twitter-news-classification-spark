import os, sys


python_version = sys.version_info.major
if python_version == 3:
    import configparser
else:
    import ConfigParser as configparser


if __name__ == '__main__':
    config = configparser.RawConfigParser()
    config.read('../config.properties')

    # Path for spark source folder
    SPARK_HOME = config.get('Spark', 'SPARK_HOME')
    os.environ['SPARK_HOME'] = SPARK_HOME

    # Append pyspark to Python Path
    sys.path.append(SPARK_HOME + config.get('Spark', 'python_path'))
    sys.path.append(SPARK_HOME + config.get('Spark', 'py4j_path'))

    from pyspark.streaming import StreamingContext
    from pyspark import SparkContext

    print('Begin streaming')
    sc = SparkContext(appName="BatchStreamingTest")
    ssc = StreamingContext(sc, 20)
    dest = config.get('Path', 'streaming_data')
    dataFile = ssc.textFileStream(dest)
    dataFile.pprint()
    ssc.start()
    ssc.awaitTermination()