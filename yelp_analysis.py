# coding: utf-8
from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct
from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.classification import LogisticRegression,\
    DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import regexp_replace
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF
import random

spark = SparkSession.builder \
    .master('local[*]') \
    .appName('yep_analysis_application') \
    .config("spark.executor.memory", "64g") \
    .config("spark.driver.memory", "64g") \
    .getOrCreate()


def valueToCategory(value):
    '''
    An auxiliary function to covert from the integer stars
    ratings to our definition of good, bad and neural
    '''
    if value == 1.0 or value == 2.0:
        return 'bad'
    elif value == 4.0 or value == 5.0:
        return 'good'
    else:
        return 'neural'


def evaluateModel(train_data, test_data, model_sr="lr"):
    '''
    Method to evaluate three different models: lr LogisticRegression
    tree for a DecisionTree and rf for a random forest. Default lr
    '''
    evaluator = BinaryClassificationEvaluator()
    if model_sr == "lr":
        lr = LogisticRegression()
        lrModel = lr.fit(train_data)
        predictions_lr = lrModel.transform(test_data)
    # predictions_lr.groupBy("label", "prediction").count().show()
        print("Test Area Under ROC for LogisticRegression: " +
              str(evaluator.evaluate(predictions_lr,
                  {evaluator.metricName: "areaUnderROC"})))
    if model_sr == "tree":
        tree = DecisionTreeClassifier()
        treeModel = tree.fit(train_data)
        predictions_tree = treeModel.transform(test_data)
        print("Test Area Under ROC for DecisionTree: " +
              str(evaluator.evaluate(predictions_tree,
                  {evaluator.metricName: "areaUnderROC"})))
    if model_sr == "rf":
        rf = RandomForestClassifier()
        rfModel = rf.fit(train_data)
        predictions_rf = rfModel.transform(test_data)
        print("Test Area Under ROC for Random Forest: " +
              str(evaluator.evaluate(predictions_rf,
                  {evaluator.metricName: "areaUnderROC"})))


def preprocess_text_data(df_with_cat,num_features):
    '''
    Transform the data in the text attribute
    We will be doing the following: remove punctuation and numbers, tokenize, 
    remove stop words, applying hashing trick and convert to TF-IDF
    '''
  
    # Remove punctuation and numbers
    df_with_cat = df_with_cat.withColumn('text', regexp_replace(df_with_cat.text, '[_():;,.!?\\-]', ' '))
    df_with_cat = df_with_cat.withColumn('text', regexp_replace(df_with_cat.text, '[0-9]', ' '))
    # Merge multiple spaces
    df_with_cat = df_with_cat.withColumn('text', regexp_replace(df_with_cat.text, ' +', ' '))
    df_with_cat = df_with_cat.withColumnRenamed("words", "temp_words")
    # Split the text into words
    df_with_cat = Tokenizer(inputCol='text', outputCol='words').transform(df_with_cat)

    locale = spark.sparkContext._jvm.java.util.Locale
    locale.setDefault(locale.forLanguageTag("en-US"))
    # Remove stop words.
    df_with_cat = StopWordsRemover(inputCol="words", outputCol="terms")\
        .transform(df_with_cat)

    # Apply the hashing trick
    df_with_cat = HashingTF(inputCol="terms", outputCol="hash", numFeatures=num_features)\
        .transform(df_with_cat)

    # Convert hashed symbols to TF-IDF
    df_with_cat = IDF(inputCol="hash", outputCol="text_features")\
        .fit(df_with_cat).transform(df_with_cat)

    return df_with_cat


def process_dataset_evaluate(df_with_cat, num_text_features, use_subjective):
    '''
    Assumes the dataset is ready afte merging and filtering.
    Herein, we will process the text data and run models.
    Two parameters: number of features and whether
    to use subjective features (cool, funny, and useful)
    '''

    df_with_cat = preprocess_text_data(df_with_cat, num_text_features)
    if use_subjective == True:
        assembler = VectorAssembler(inputCols=['cool', 'funny', 'useful', 'text_features'], outputCol='features')
    else:
        assembler = VectorAssembler(inputCols=['text_features'], outputCol='features')
    
    df_with_cat= assembler.transform(df_with_cat)
    # Specify a seed for reproducibility
    df_with_cat_train, df_with_cat_test = df_with_cat.randomSplit([0.8, 0.2], seed=23)
    # To see the statistics of the data (Slide 12 presentation) uncomment the following
    # df_with_cat_train.groupBy('label').count().show()
    # df_with_cat_test.groupBy('label').count().show()

    # Run some models
    evaluateModel(df_with_cat_train, df_with_cat_test, "lr")
    evaluateModel(df_with_cat_train, df_with_cat_test, "tree")
    evaluateModel(df_with_cat_train, df_with_cat_test, "rf")

# Load the datasets to be used in our experiments
business_data = spark.read.json('business.json')
review_data = spark.read.json('review.json')
# Preprocessing and merging 
print("Business schema")
business_data.printSchema()

business_data = business_data.select('business_id','name','categories','city','state')

print("Filtered business schema")
business_data.printSchema()
print("Review data schema")
review_data.printSchema()
review_data = review_data.select('business_id','cool','funny','useful','review_id','stars','text')
review_data = review_data.withColumnRenamed('business_id','b_id')
print("Review filtered data schema")
review_data.printSchema()

# We will study only US states and territories
us_states_territories = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
          "AS", "DC", "FM", "GU", "MH", "MP", "PW", "PR", "VI"]
us_states_territories_df = spark.createDataFrame(us_states_territories, StringType())
bussines_us = business_data.join(us_states_territories_df,business_data.state == us_states_territories_df.value)
count_business_us = bussines_us.count()
print("count business us ",count_business_us )
# Found null elements in categories drop them
# bussines_us.filter('categories is null').count()
bussines_us = bussines_us.na.drop(subset=['categories'])
restaurants_us = bussines_us.select('*').where("categories like '%Restaurant%' or categories like '%restaurant%' ")
restaurants_us = restaurants_us.drop('value')
# Let us join our tables
reviews_restaurants_us = review_data.join(restaurants_us,restaurants_us.business_id == review_data.b_id)
udfValueToCategory = udf(valueToCategory, StringType())
df_with_cat = reviews_restaurants_us.withColumn("label", udfValueToCategory("stars"))
df_with_cat = df_with_cat.withColumnRenamed('label','label_text')
df_with_cat = StringIndexer(
inputCol="label_text",
outputCol="label"
).fit(df_with_cat).transform(df_with_cat)
# Last, we filter bad and good reviews only
df_with_cat = df_with_cat.select('*').where(" label_text = 'bad' or label_text ='good' ")
# Data is ready for text preprocessing and model testing
print("Now running the models")
# Slide 13 of the presentation
process_dataset_evaluate(df_with_cat, 5000, True)
process_dataset_evaluate(df_with_cat, 5000, False)
# Correlation analysis after filtering (Slide 14 of the presentation)
print("Correlation of the latent subjective attributes with stars") 
print(df_with_cat.stat.corr('funny','stars'))
print(df_with_cat.stat.corr('useful','stars'))
print(df_with_cat.stat.corr('cool','stars'))
spark.stop()

