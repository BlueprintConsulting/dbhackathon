# Databricks notebook source
# MAGIC %md # Attribution Modeling

# COMMAND ----------

# MAGIC %md 
# MAGIC #### The idea is a fairly simple one. You have several customers that made one or more purchases of your products, and you want to know which of your marketing channels were worth the investment based on which ones these customers encountered. It seems that it should be as straightforward as looking at all of the channels that they engaged with. But this raises some questions. First, how much weight should you apply to each of the channels? Should the most weight be applied to the first interaction with the customer, or the last before they made their final purchase? What if two customers engaged with all of the same channels, but in a different order? Is that relevant? These are exactly the kind of questions one is trying to answer with attribution modeling.
# MAGIC #### Applying data science to marketing is an idea that seems to have only gained significant recognition in the last 4-5 years. While the idea of attribution modeling has been around ever since marketing first met statistics, the complexity and theory behind it has grown exponentially since data science was initially put into use to solve the problem. Ranging from a simple first-touch model to time-series forecasting and even neural networks, data scientists are using every tool they know how to get the best results from an attribution model, and with good reason. The more that the marketing team knows about what is working and what isn't, the better position they will be in the maximize their ROI and create better efficiency for the company's future campaign strategies.
# MAGIC #### In this notebook we will go over the basics of attribution modeling as it is applied to digital marketing.

# COMMAND ----------

# MAGIC %md ## Capabilities and Limitations Associated With Attribution Modeling

# COMMAND ----------

# MAGIC %md ### What it can do
# MAGIC * #### Attribution modeling can use observational data to make reasonable estimates of ROI on indivudal ad channels, content types, etc. in order to maximize campaign effectiveness, or otherwise.
# MAGIC * #### It can provide insights into how revenue would be effected by an increase or decrease in campaign funding or intensity.
# MAGIC 
# MAGIC ### What it can't do
# MAGIC * #### Attribution modeling cannot provide causal inference. In other words, it can't tell you (with certainty) that X dollars spent on Y media channel will produce Z dollars in revenue. That is a job for A/B testing.

# COMMAND ----------

# MAGIC %md ## Common Challenges

# COMMAND ----------

# MAGIC %md * #### Depending on the analytic approach and age of the company, a lack of data can be a serious issue.
# MAGIC * #### Accounting for selection bias, which is the correlation between an unobservable demand variable and the model variables. For example, targeting a population with ads that has previously shown interest in your product.
# MAGIC * #### Collecting and aggregating exposure data from all of the different avenues that advertising usually uses can be difficult and expensive.
# MAGIC * #### Correlated variables caused by not enough variability in advertising levels or otherwise make statistical analysis unpredictable.

# COMMAND ----------

# MAGIC %md ## Single-Source Attribution

# COMMAND ----------

# MAGIC %md * #### First-touch Attribution: All of the credit is assigned to the first campaign that the customer engaged with, ignoring everything that came afterwards.
# MAGIC * #### Last-touch Attribution: All credit is assigned to the campaign that the customer engaged with just prior to making the purchase. Everything before the "last touch" is given 0 credit.
# MAGIC 
# MAGIC #### Since a typical customer in today's market can engage with several different campain channels, and any of them can lead to the final sale, these models tend not to provide very accurate results and are seldom used anymore.

# COMMAND ----------

# MAGIC %md ## Multi-Source Attribution

# COMMAND ----------

# MAGIC %md #### There are several options for multi-source (or multi-touch) attribution modeling, which account for more than one touchpoint along the customer journey. The differences in the following models come from the weights that are applied to each of the touchpoints. Here are the three most interesting multi-source attribution models:
# MAGIC 
# MAGIC * #### Time Decay: The closer the touchpoint is to the final close, the more weight it is given. This model assigns little importance to how the customer initially engaged with the business, with most of the emphasis on the end of the customer journey.
# MAGIC * #### W-shaped: This model assigns more weight to three major touchpoints along the customer journey. These points are the first touch, the lead creation, and the opportunity creation. Typically, these three touchpoints are each given 30% of the weight, while the remaining 10% is split between the remaining touchpoints.
# MAGIC * #### Full Path: Similar in design to the w-shaped model, this model covers the entire customer journey, through the final close. Extra weight is assigned to all of the major touchpoints, those from the w-shaped model and the final close touchpoint, while the remaining get an equal amount of the leftover credit. The main advantage of this model is that it accounts for the entire sales cycle, including the engagement that comes after the opportunity is created.
# MAGIC 
# MAGIC #### The advantages of these models are that they are still fairly simple, easy to implement and understand, while at the same time accounting for more of the complexity that is involved in the customer sales cycle. One serious disadvantage is that the weights are stagnant. Meaning the weights are assigned often with only minimal knowledge of the data and are not representative of the customers themselves. The model and weights used here do not vary depending on the data as there is no statistical or machine learning involved. These are good initial models to run for the purpose of gaining some fundamental insights about the marketing data and how best to move forward from there.

# COMMAND ----------

# MAGIC %md ## Attribution Modeling Exercise

# COMMAND ----------

# Import some necessary packages for data collection and manipulation.
from pyspark.sql.functions import udf
from pyspark.sql.types import *
import pyspark

# COMMAND ----------

# MAGIC %md
# MAGIC # Get the Data
# MAGIC 
# MAGIC ### Source Data
# MAGIC 
# MAGIC There are multiple data sources that need to be integrated.  They can all be found in a public Azure Blob account located here: `wasbs://public@hackathonpublic.blob.core.windows.net`
# MAGIC 
# MAGIC [Connecting to Azure Data Sources](https://docs.azuredatabricks.net/spark/latest/data-sources/index.html)
# MAGIC 
# MAGIC Here is a description of the different datasets (and their paths in the Azure Blob account listed above):
# MAGIC 
# MAGIC * **Purchase History (Parquet)** `/attribution-modelling/data2/purchases` - a log of historical purchases by each user
# MAGIC * **Marketing Touchpoint History (Parquet)** `/attribution-modelling/data2/events-*` - a log of historical marketing touchpoints by user (there are actually 3 of these tables, one for each of 3 marketing channels)
# MAGIC * **ID Maps (JSON)** `/attribution-modelling/data2/id-maps` - mapping of user id (from purchase history) to user uuid's found in the event tables (there is a mapping table for each of the 3 marketing channels)
# MAGIC 
# MAGIC ### Transforming the Data into something usable
# MAGIC 
# MAGIC You will need to read in these datasets, and manipulate them such that you have a table where rows correspond to discrete purchase events by a user, and has these columns (or something similar):
# MAGIC 
# MAGIC * `user` - the user id
# MAGIC * `purchaseDate` - the date of this purchase
# MAGIC * `customerJourney` - the marketing touchpoints that preceeded this purchase represented as an array (but be careful, many users have multiple purchases)
# MAGIC * `purchaseAmount` - the total spent on this purchase

# COMMAND ----------

# Configure Spark to connect to an Azure Blob Storage Account
spark.conf.set("fs.azure.account.key.hackathonpublic.blob.core.windows.net","obZL8SDLs39pjQ0xyO41uxVMqHkoAHmoSzmZUjB8HEEpg2DFGt0T00WGBbzmcK9rZU7vRwqbnZc2RrcNGGLmNg==")

# Example of creating a Dataframe from Parquet and validating the filepath
purchase_df = spark.read.parquet("wasbs://public@hackathonpublic.blob.core.windows.net/attribution-modelling/data2/purchases")
dbutils.fs.ls("wasbs://public@hackathonpublic.blob.core.windows.net/attribution-modelling/data2/purchases")

# YOUR CODE HERE

# END OF YOUR CODE BLOCK

# COMMAND ----------

# The data frame consist of four columns:
# User: IntergerType, User ID for a particular customer.
# Purchase Date: StringType, The date that the customer made a purchase (one per line).
# Path: ArrayType, The marketing channels that the customer "touched" between their last purchase (if there was one) and this one. The three possible
# marketing channels include paid_search, social_media, and marketing_email.
# Purchase Amount: DoubleType, The amount spent by the customer on this purchase.
attribution_model_df.show()

# COMMAND ----------

# First, let's do some exploratory data analysis.
# Looking at the plot below, it is clear that the sales for this company follow an approximate sine curve showing peak sales
# in the early spring and early fall, with fewer sales in the summer and winter months.display
display(attribution_model_df.groupBy('Purchase Date').count())

# COMMAND ----------

# In the below histogram, we can see that the average customer tends to spend between 200 and 500 dollars in a single purchase.
display(attribution_model_df.select('Purchase Amount'))

# COMMAND ----------

# Create plots to show the probability of a customer interacting with touchpoint B after touchpoint A. In other words, after a customer
# has interacted with a paid search ad, how likely are they to interact with a social media ad right afterwards? Then, create a plot to 
# show the most common paths that customers take.

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC #### First-Touch Attribution

# COMMAND ----------

# Create a first-touch attribution model by looking at only the first touchpoint in a customer's path and crediting all of the purchase
# amount to that marketing channel.

# COMMAND ----------

first_touch = spark.createDataFrame(attribution_model_df.select('Path').rdd.map(lambda x: pyspark.sql.Row(x[0][0]) if len(x[0]) >= 1 else pyspark.sql.Row('direct_purchase')).collect(), ['FirstTouch'])

# COMMAND ----------

# Here is a histogram of first touches by purchase.
display(first_touch.groupBy('FirstTouch').count().sort('FirstTouch'))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC #### Last-Touch Attribution

# COMMAND ----------

# Create a last-touch attribution model by looking at only the last touchpoint in a customer's path and crediting all of the purchase
# amount to that marketing channel.

# COMMAND ----------

last_touch = spark.createDataFrame(attribution_model_df.select('Path').rdd.map(lambda x: pyspark.sql.Row(x[0][-1]) if len(x[0]) >= 1 else pyspark.sql.Row('direct_purchase')).collect(), ['LastTouch'])

# COMMAND ----------

# Here is a histogram of last touches by purchase. Notice any differences between this graph and the first-touch graph above.
display(last_touch.groupBy('LastTouch').count().sort('LastTouch'))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# Before moving on, compare your results of the last-touch model with the results of the first-touch model. Which do you trust more? Why?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Multi-Source Attribution

# COMMAND ----------

# While there are a few options available for the multi-touch model, research has shown that the touchpoint closest to the time of purchase
# tends to be the most relevant. For the next section, use the time decay function defined below to attach the most credit to the last
# touchpoint, with exponentially less credit attached to the touchpoints before it. Then, compare this model to the two models above. How
# do they differ?

# COMMAND ----------

# MAGIC %md
# MAGIC $$ y = e^{-\alpha x} $$

# COMMAND ----------

# Start with alpha = 1 and then experiment with different values of alpha.

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Validation

# COMMAND ----------

# Validating you model is the most important, and often most difficult, part of attribution modeling. So our model tells us that social
# media ads deserve most of the credit. Why should we beleive that? While there isn't always a "right" way to validate the model, there
# are some things we can do to help build confidence in the information we have collected. 
# First, A/B testing to see whether the changes we incorporate differ significantly from a control group.
# Second, we could gather for information from domain experts in our particular industry.
# Third, a survey could be conducted to ask customer which type of ad most contributed to their purchase.
# Lastly, there are several other analytical models including media mix modeling and time-series forecasting that we could explore.

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


