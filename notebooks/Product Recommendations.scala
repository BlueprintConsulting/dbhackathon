// Databricks notebook source
// MAGIC %md 
// MAGIC  - Using Azure Databricks with SparkSQL
// MAGIC  - Using Azure Databricks with Spark Dataframes for Text Analysis
// MAGIC  - Using Azure Databricks to Recommend Items%md 

// COMMAND ----------

spark.conf.set(
  "fs.azure.account.key.marcdatabricksdemo.blob.core.windows.net",
  "npCy5fGSsUsKNDyVha3vwUWyrHzlmqeKSMgWbQO3XeZL2iqghEr9e5b0+vkbewnPqCQ7RkwnedOQkkE1KZP3kQ==")

val df = spark.read.parquet("wasbs://reviews@marcdatabricksdemo.blob.core.windows.net/reviews/")

df.write.saveAsTable("demoreviews")

// COMMAND ----------

// MAGIC %md **productId is the Key**

// COMMAND ----------

// MAGIC %md **1. Create 2 datasets; one of distinct brands w/ productId's and distinct titles (called items) w/ productId's. Make sure to drop nulls as well**

// COMMAND ----------

val brands = 

// COMMAND ----------

val items = 

// COMMAND ----------

// MAGIC %md **2. Take a look at items where productId is B000HCR8C4**

// COMMAND ----------

display(items.where("productId = 'B000HCR8C4'"))

// COMMAND ----------

// MAGIC %md **3. Save the Brand and items Datasets as Tables**

// COMMAND ----------



// COMMAND ----------

// MAGIC %md **4. Join Items and Brands and take a look at what item and brand has the same productId**

// COMMAND ----------



// COMMAND ----------

// MAGIC %md **5. Find which brands make up the most elements in the combination of Brands and Items table**

// COMMAND ----------



// COMMAND ----------

// MAGIC %md ##Using Azure Databricks with Spark Dataframes for Text Analysis

// COMMAND ----------

// MAGIC %md **6. Make a dataFrame of the joined Items & Brands Tables**

// COMMAND ----------

val itemBrands = 

// COMMAND ----------

itemBrands.cache().take(100)

// COMMAND ----------

// MAGIC %md **7. Break the text in the titles into individual words in the same column** (Hint: You'll need to use a transformer)

// COMMAND ----------

import org.apache.spark.ml.feature._
import org.apache.spark.ml._


val tokenizer = 

// COMMAND ----------

// MAGIC %md **8. Apply the transformer and display the split column**

// COMMAND ----------

val keywords = 

// COMMAND ----------

display(keywords)

// COMMAND ----------

// MAGIC %md **9. Get a count of Keywords by Brands**

// COMMAND ----------

import org.apache.spark.sql.functions._


// COMMAND ----------

display(keywordsByBrand)

// COMMAND ----------

// MAGIC %md **10. Which key words are most popular for the Disney brand?**

// COMMAND ----------

 

// COMMAND ----------

// MAGIC %md **11. What about Apple**

// COMMAND ----------



// COMMAND ----------

// MAGIC %md ## Using Azure Databricks to Serve Machine Learning Models

// COMMAND ----------

// MAGIC %sql select * from demoreviews

// COMMAND ----------

// MAGIC %md ![Alternating Least Squares - Matrix Factorization](https://raw.githubusercontent.com/cfregly/spark-after-dark/master/img/ALS.png)

// COMMAND ----------

// MAGIC %md **12. We'll need to use Datasets, so create a dataset from our 'demoreviews' table. <br/>
// MAGIC Then take a look at the Brand, Title, and Ratings for our user 'A3OXHLG6DIBRW8'**

// COMMAND ----------

val dataset =

// COMMAND ----------

// MAGIC %md **13. Let's train our recommendation model using ALS**
// MAGIC 1. Register a UDF called HashID using ("generateHashCode", (s : String) => s.hashCode) in the UDF to get a HashID Function
// MAGIC 2. Create a new training dataset by applying the hashIF UDF to the productId (call it itemID) column and the User (call it userID) column
// MAGIC 3. Create a model by using ALS() and setting the item columns to "itemID" and user column to "userID"
// MAGIC 4. Fit the model to the training data

// COMMAND ----------

import org.apache.spark.ml.recommendation._

val dataset = table("demoreviews")
val hashId = sqlContext.udf.register("generateHashCode", (s : String) => s.hashCode)
val trainingData = 
val als = new ALS().setItemCol("").setUserCol("") 
val model = 

// COMMAND ----------

display(trainingData)

// COMMAND ----------

// MAGIC %md **14. Create a dataframe from the combined Item and Brands Dataframe from above for user "A3OXHLG6DIBRW8" and make sure to create columns itemId and UserId using the UDF from above**  <br/>

// COMMAND ----------

val userItems =

// COMMAND ----------

display(userItems)

// COMMAND ----------

// MAGIC %md **15. Apply the recommendation to the userItems table and display the top 10 results by prediction** 

// COMMAND ----------

val recommendations = 

// COMMAND ----------

display(recommendations.select("prediction", "productId", "title", "brand").orderBy(desc("prediction")))

// COMMAND ----------

display(itemBrands.join(recommendations, "productId").orderBy(desc("prediction")).limit(10))

// COMMAND ----------

// MAGIC %md ###Extra Credit: Improve our model by using a: parameter grid, Regression evaluator for the error, in a function called TrainValidationSplit

// COMMAND ----------

https://www.youtube.com/watch?v=FgGjc5oabrA
