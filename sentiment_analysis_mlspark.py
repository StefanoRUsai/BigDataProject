#import findspark
#findspark.init()

"""
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

conf = SparkConf().setAppName("Sent").setMaster("local[*]").set('spark.driver.memory','4g')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
"""
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import udf
from sklearn.metrics import classification_report, confusion_matrix
import string
from time import time, clock

def remove_punct(text):
	return text.translate(str.maketrans('', '', string.punctuation))

def map_value(emotions):
    if emotions == 1:
        return 'anger'
    elif emotions == 2:
        return 'fear'
    elif emotions == 3:
        return 'sadness'
    elif emotions == 4:
        return 'joy'
    elif emotions == 5:
        return 'surprise'
    elif emotions == 6:
        return 'love'

if __name__ == '__main__':
	start = clock()
	appName = "Sent"
	spark = spark = SparkSession.builder.appName("Big").getOrCreate()

	tweets_csv = spark.read.option("inferSchema", "true").option("delimiter", ";").csv('/home/ubuntu/dataset.csv', inferSchema=True, header=True).orderBy(rand()).repartition(2)
	tweets_csv.show(truncate=False, n=10)

	tweets_csv.columns


	tweets_csv.count()

	punct_remove = udf(lambda s: remove_punct(s))


	data_df = tweets_csv.withColumn("text", punct_remove(tweets_csv["text"]))



	tokenizer = Tokenizer(inputCol="text", outputCol="words")
	data_df = tokenizer.transform(data_df)

	stopwords = StopWordsRemover(inputCol="words", outputCol="filtered")
	data_df = stopwords.transform(data_df)



	data_df.show()



	hashing_tf = HashingTF(inputCol='filtered', outputCol='raw_features')
	data_df = hashing_tf.transform(data_df)

	idf = IDF(inputCol="raw_features", outputCol="features")
	idf_model = idf.fit(data_df)
	data_df = idf_model.transform(data_df)



	data_df.show()



	train_df, test_df = data_df.randomSplit([0.75, 0.25])



	test_df.show()
	train_df = train_df.withColumnRenamed("emotions", "label")
	test_df = test_df.withColumnRenamed("emotions", "label")

	lr = LogisticRegression(featuresCol="features", labelCol="label")
	model = lr.fit(train_df)
	print('####### primo train finito #######')

	predict_train = model.transform(train_df)
	print('####### transform ###########')

	end=start - clock()
	print(f'Secondi totali {end}')

	y_true = predict_train.select(['label']).collect()
	y_pred = predict_train.select(['prediction']).collect()
	print('####### y_true, y_pred select ###########')
	end=start - clock()
	print(f'Secondi totali {end}')
	predict_test = model.transform(test_df)

	predict_test.select("label", "prediction").show(10)

	evaluation = model.evaluate(test_df)



	print(f'accuracy: {evaluation.accuracy}')
	print(f'precisionByLabel: {evaluation.precisionByLabel}')
	print(f'recallByLabel: {evaluation.recallByLabel}')


	predictionAndTarget = predict_train.select("label", "prediction")

	evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
	acc = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "accuracy"})
	f1 = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "f1"})
	weightedPrecision = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedPrecision"})
	weightedRecall = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedRecall"})
	print(f'accuracy: {acc}')
	print(f'f1: {f1}')
	print(f'weighted Precision: {weightedPrecision}')
	print(f'weighted Recall: {weightedRecall}')

	paramGrid = ParamGridBuilder() \
	    .addGrid(hashing_tf.numFeatures, [100]) \
	    .addGrid(lr.regParam, [0.01]) \
	    .build()
	evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

	cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
	print('################################### CROSS VALIDATION ##############################################')
	cvModel = cv.fit(train_df)
	predict_train=cvModel.transform(train_df)

	print('################################### Transform CROSS VALIDATION ##############################################')

	predict_test=cvModel.transform(test_df)
	print(f"L'area sotto la curva ROC per il train set dopo la cross validation è {evaluator.evaluate(predict_train)}")
	print(f"L'area sotto la curva ROC per il test set dopo la cross validation è {evaluator.evaluate(predict_test)}")

	end=start - clock()
	print(f'Secondi totali {end}')
