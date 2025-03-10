![cover_photo](../docs/readme_files/Online_Shoppers.jpg)

# Customer Segmentation For An Online Retail Store

*Customer Segmentation is a valuable tool for businesses, enabling them to align their strategies and tactics more effectively with current and potential customers. In today's marketing landscape, whether in B2B (business-to-business) or B2C (business-to-consumer) contexts, companies segment their customers based on various factors, including demographic, geographic, behavioral, psychographic, and technographic characteristics.The methodologies for customer segmentation are constantly evolving and may differ from one company to another. However, the primary goal of effective customer segmentation analysis is to gain a better understanding of customers' ever-changing needs and behaviors.*

*RFM (Recency, Frequency, Monetary) Analysis is a powerful method that enables marketing teams to create targeted campaigns for specific customer segments. These campaigns include offers, messages, and services that are relevant and tailored to customers’ buying patterns and behaviors.*

*An effective customer segmentation analysis leads to better customer insights, which enable marketing teams to make informed strategic decisions regarding messaging and positioning. This strategic approach fosters innovative and creative solutions that enhance profitability.* 

*This project aims to utilize machine learning for customer segmentation based on RFM analysis, focusing on three key features: Recency, Frequency, and Monetary value. The analysis aims to uncover valuable insights into customer purchasing behavior and transform the transactional data into a customer-centric dataset through feature engineering and clustering that will effectively aid in segmenting customers, enabling the business to determine appropriate marketing strategies and enhance the customer experience. Additionally, this analysis has the potential to significantly boost product sales, offering a promising outlook for the future*

## 1. Data

Data Source: [UCI Machine Learning Repository | Online Retail II]('https://archive.ics.uci.edu/dataset/502/online+retail+ii')

The dataset contains all transactions for a UK-based online retailer that sells all occasion gift-ware. This e-commerce dataset, made available by the UCI Machine Learning Repository, contains transactions made by customers from 2009 to 2011. This project will work on the latest transactions done in 2010 by approximately 4,300 customers.

Detailed description of the fields can be found in the aforementioned link.

## 2. Data Wrangling

The [Data Wrangling step](../notebooks/DataWrangling.ipynb) kicked off with 541,910 records in the 2010 tab of the dataset. There were inconsistencies found in the dataset such as blank customer ID (~25%), negative quantities (1.6%) which were found to be related to cancelled invoices, and duplicate records (~1% ). These records were dropped from the dataset as they were deemed not helpful in analyzing the customer transactions. 

An interesting column is the invoices which have values not conforming with the documented notation; however, thorough investigation suggested thatthose records don't contain irregularities in their respective numeric data. An assumption was then made that these transactions were made manually or done outside the order placement process. Because there are no anomalies in the numeric data, these records were not dropped.

A total of 392733 records remain at the end of the Data Cleaning step.

![Data Cleaning Summary](../docs/readme_files/Summary_DW.png)

## 3. EDA

The [EDA step](../notebooks/ExploratoryDataAnalysis.ipynb) revealed 0-priced items and transactions later than December 30, 2010. These records were then removed and the dataset ended up with 367,023 records after dropping those invalid records. Out of the 367,023 records the valid transactions for the whole year can be summarized as follows:

- Unique Invoices:  17132
- Unique Countries:  36
- Unique StockCodes:  3596
- Unique Customer IDs:  4219

Exploratory Data Analysis of the monthly sales indicated a steep decline in sales in December was revealed.

![EDA monthly sales](..docs/readme_files/EDA_monthlysales.png)

Further investigation revealed that the transactions for December was incomplete and that the latest was captured on the 9th of December which even less than the first half of the month. The total transactions in December only accounted for less than 5% of the whole transactions for the year. 
Although there was transaction as high as 168469.6, much of the sales during the month are in the lower amount. 

![December sales distribution](../docs/readme_files/EDA_decemberdist.png)


## 4. Pre-processing

[Preprocessing](../notebooks/Preprocessing.ipynb) step began by feature engineering the columns to reflect the RFM features per customer - that is:

- Recency - How long has it been since the customer's last purchase date? * 
- Frequency - How many transactions in did the customer have? 
- Monetary - How much was spent by the customer?

The distribution of data per feature is right-skewed, which indicates presence of outliers.

![RFM Distribution](../docs/readme_files/FE_rfmdist.png)

The focus of this project is on clustering, so only the non-outliers will be processed by the model. 
It is necessary to perform separate analysis on outliers as they represent extreme behaviours by the customers, such as very big spending and very frequent purchases. Below is the boxplot before and after separating the outliers:

![RFM Outliers](../docs/readme_files/FE_outlierdist.png)

These outliers will be included in the cluster analysis down the line:

- Monetary outliers:  402
- Frequency outliers:  412

## 5. Modeling

3 clustering algorithms have been considered for the analysis: 
- KMeans Clustering 
- Agglomerative Clustering 
- DBSCAN

In the [Modeling](../notebooks/Modeling.ipynb) step, three hyperparameter optimization searches were tested accross all three aforementioned algorithms by utilizing the nested cross validation approach to determine the optimal parameters per algorithm. As the process cycles through each algorithm, a scoring function is used to determine the silhouette score for each algorithm. In the results below, Silhouette scores were computed for Agglomerative Clustering and DBSCAN due to their lack of loss function. 

SMBO (Sequential Model-Based Optimization) can't be used on AC and DBSCAN as they don't have a loss function. Although SMBO was used in KMeans alone, the scoring function (Silhouette score) can't be minimized so the result was deliberately tagged as 'Score N/A'.

|Algorithm        |Tuning            |Best_params                                                                                                                                                 |Silhouette_Score    |
|-------------|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
|KMeans       |GridSeachCV       |{'cluster__algorithm': 'lloyd', 'cluster__init': 'k-means++', 'cluster__max_iter': 100, 'cluster__n_clusters': 2, 'cluster__n_init': 5, 'cluster__tol': 0.1}|0.602058462575161   |
|KMeans       |RandomizedSearchCV|{'cluster__tol': 0.1, 'cluster__n_init': 10, 'cluster__n_clusters': 2, 'cluster__max_iter': 200, 'cluster__init': 'random', 'cluster__algorithm': 'lloyd'}  |0.5984635762963102  |
|Agglomerative|GridSeachCV       |{'cluster__compute_distances': True, 'cluster__linkage': 'complete', 'cluster__metric': 'euclidean', 'cluster__n_clusters': 2}                              |0.489265330024139   |
|Agglomerative|RandomizedSearchCV|{'cluster__n_clusters': 2, 'cluster__metric': 'manhattan', 'cluster__linkage': 'single', 'cluster__compute_distances': True}                                |0.3580704774506997  |
|DBSCAN       |RandomizedSearchCV|{'cluster__min_samples': 10, 'cluster__metric': 'euclidean', 'cluster__eps': 0.16999999999999998}                                                           |-0.15949935617142658|
|DBSCAN       |GridSeachCV       |{'cluster__eps': 0.16999999999999998, 'cluster__metric': 'euclidean', 'cluster__min_samples': 10}                                                           |-0.15949935617142658|
|KMeans       |SMBO              |{'algorithm': 1, 'init': 0, 'max_iter': 991.0, 'n_clusters': 2.0, 'n_init': 6.0, 'tol': 3}                                                                  |Score N/A           |

Out of the model evaluation outcomes, **KMeans** and **Agglomerative Hierarchichal** clustering yielded more promising silhouette scores. It can also be surmised that DBSCAN may not be the appropriate clustering algorithm for the dataset since it has negative silhouette scores which indicates that the clustering was poor and overlap more compared to other algorithms that have higher silhouette score. 

Determining the best parameters for KMeans using GridSearchCV, RandomizedSearchCV, and SMBO with the inertia as the criteria produced the following results:

|Algorithm        |Tuning            |Best_params                                                                                                                                                 |Inertia             |
|-------------|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
|KMeans       |SMBO              |{'algorithm': 0, 'init': 1, 'max_iter': 197.0, 'n_clusters': 2.0, 'n_init': 9.0, 'tol': 2}                                                                  |6419.7229907037645  |
|KMeans       |GridSearchCV      |{'cluster__algorithm': 'lloyd', 'cluster__init': 'k-means++', 'cluster__max_iter': 100, 'cluster__n_clusters': 2, 'cluster__n_init': 5, 'cluster__tol': 0.1}|5798.164356724086   |
|KMeans       |RandomizedSearchCV|{'cluster__tol': 0.1, 'cluster__n_init': 10, 'cluster__n_clusters': 2, 'cluster__max_iter': 200, 'cluster__init': 'random', 'cluster__algorithm': 'lloyd'}  |5780.09743041892    |

GridSearch and Random search methods are relatively inefficient compared to SMBO.
SMBO works by considering the previously seen hyperparameter combinations when choosing the next set of hyperparameters to evluate. Grid and random searches, on the other hand, are completely uninformed by past evaluations and spends significant amount of time evaluating “bad” hyperparameters.

The best parameters from SMBO were used in re-building the model.

**Chosen algorithm : KMeans**
**Best parameters : {'algorithm': 'lloyd', 'init': 'k-means++', 'max_iter': 197, 'n_clusters': 2, 'n_init': 9, 'tol': 0.01}**

To get a sense of the best K- no. of clusters, an inertia plot was shown:

![Inertia Plot](../docs/readme_files/Modeling_inertia_plot.png)

The "knee" point could is either n_clusters=3 or n_clusters=4 so an additional Silhouette Analysis was performed. The Silhouette Plot revealed that between n_clusters=3 and n_clusters=4, the former has lower negative silhouette coefficient values, and the cluster label heights is even better compared to the latter.

![Inertia Plot](../docs/readme_files/Modeling_silplot.png)


**The optimal k number of clusters for this data using KMeans algorithm is 3.**

## 6 Cluster Analysis

Meaningful labels can be assigned by looking at the distribution of the clusters in terms of the three key features - recency, frequency, and monetary values. 

![Clusters Distribution](../docs/readme_files/Cluster_violinplot.png)
 
Cluster 0: **Moderate**
- Moderately frequent buyers that are not necessarily high spenders, and haven't purchased recently. 
Cluster 1: **Recent**
- Less frequent buyers who are low-spenders but made recent purchases.
Cluster 2: **Loyal**
- Frequent shoppers who are high spenders, although no recent purchases.

Outliers in the data are designated as follows:
>  - Monetary outliers : High-Spenders Customers
>  - Frequency outliers : Frequent Customers
>  - Both Monetary and Frequency outliers : VIPs 

### CUSTOMER SEGMENTS

![Customer Segments Bar](../docs/readme_files/Cluster_finalbar.png)
![Customer Segments Tree](../docs/readme_files/Cluster_treemap.png)

<font color='#fda848'><b>MODERATE</b></font>
- Moderately frequent buyers that are not necessarily high spenders, and haven't purchased recently. 

Recommendation: 
1. Offer subscription on frequently bought items (whenever applicable)
2. Recommend "Frequently bought together" items
3. Implement customer retention and loyalty programs

<font color='#69c641'><b>RECENT</b></font>
- Less frequent buyers who are low-spenders but made recent purchases.

Recommendations:
1. Identify recently purhcased products and run targeted "similar products" ad recommendations
2. Encourage to purchase more by offering incentives, vouchers, discounts, and bundle deals. 
3. Enhance customer experience and services

<font color='#fd4848'><b>LOYAL</b></font>
- Frequent shoppers who are high spenders, although no recent purchases.

Recommendations:
1. Re-engage by implementing rewards and loyalty programs, and exclusive perks
2. Run targeted ads for trending items and top items sold 
3. Offer subscription on frequently bought items (whenever applicable)

<font color='#4196c6'><b>VIPs</b></font>
- High value, frequent buyers. 

Recommendations:
1. Offer exclusive perks, and vouchers
2. Pamper and enhance shopping experience by offering expedited or free shipping (whenever applicable)
2. Offer additional discounts on bulk purchases (whenever applicable)

<font color='#04b1b9'><b>FREQUENT</b></font>
- Very frequent buyers. 

Recommendations:
1. Offer subscription on frequently bought items (whenever applicable)
2. Implement customer retention and loyalty programs
3. Offer expedited or free shipping vouchers (whenever applicable)

