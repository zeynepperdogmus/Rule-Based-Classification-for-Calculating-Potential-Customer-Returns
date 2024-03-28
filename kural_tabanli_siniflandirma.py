#############################################
# Rule-Based Classification for Calculating Potential Customer Returns
#############################################

#############################################
# Business Problem
#############################################
# A gaming company aims to create level-based new customer profiles (personas) using some characteristics of its customers
# and wants to predict how much new customers coming to the company according to these new customer profiles can potentially
# earn for the company.

# For example: It's desired to determine how much on average a 25-year-old male user from Turkey using IOS can potentially earn.


#############################################
# Data Story
#############################################
# The Persona.csv dataset contains the prices of products sold by an international gaming company and some demographic 
# information of the users who purchased these products. The dataset consists of records generated in each sales transaction.
# This means the table is not singularized. In other words, a user with certain demographic characteristics may have made
# multiple purchases.

# Price: Amount spent by the customer
# Source: Type of device the customer connected to
# Sex: Gender of the customer
# Country: Country of the customer
# Age: Age of the customer

################# Before Implementation #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# After Implementation #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


#############################################
# PROJECT TASKS
#############################################

#############################################
# TASK 1: Answer the following questions.
#############################################


# Question 1: Read the persona.csv file and display general information about the dataset.
import pandas as pd
pd.set_option("display.max_rows", None)
df = pd.read_csv(r'\Users\ZEYNEP\OneDrive\Masaüstü\PycharmProjects\persona.csv')
df.head()
df.shape
df.info()

# Question 2: How many unique SOURCE values are there? What are their frequencies?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# Question 3: How many unique PRICE values are there?
df["PRICE"].nunique()

# Question 4: How many sales have been made for each PRICE?
df["PRICE"].value_counts()
df.groupby("PRICE").agg({"PRICE":"count"})

# Question 5: How many sales have been made from each country?
df["COUNTRY"].value_counts()

df.groupby("COUNTRY")["PRICE"].count()
df.groupby("COUNTRY")[["PRICE"]].count()
df.groupby("COUNTRY")["COUNTRY"].count()

df.pivot_table(values="PRICE",index="COUNTRY",aggfunc="count")


# Question 6: How much revenue has been earned from sales by country?
df.groupby("COUNTRY")["PRICE"].sum()
df.groupby("COUNTRY").agg({"PRICE": "sum"})

df.pivot_table(values=["PRICE"],index=["COUNTRY"],aggfunc="sum")



# Question 7: What are the sales counts by SOURCE types?
df["SOURCE"].value_counts()


# Question 8: What are the average prices by countries?
df.groupby(['COUNTRY']).agg({"PRICE": "mean"})

# Question 9: What are the average prices by SOURCE?
df.groupby(['SOURCE']).agg({"PRICE": "mean"})


# Question 10: What are the average prices by COUNTRY-SOURCE breakdown?
df.groupby(["COUNTRY", 'SOURCE']).agg({"PRICE": "mean"})


#############################################
# TASK 2: What are the average earnings by COUNTRY, SOURCE, SEX, and AGE breakdown?
#############################################
df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).head()

#############################################
# TASK 3: Sort the output by PRICE.
#############################################
# Apply the sort_values method to PRICE in descending order to better visualize the output from the previous question.
# Save the output as agg_df.

agg_df = df.groupby(by=["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
agg_df.head()



#############################################
# TASK 4: Convert the names in the index to variable names.
#############################################
# All variables except PRICE in the output of the third question are index names.
# Convert these names to variable names.
# Hint: reset_index()

agg_df.reset_index(inplace=True)
#agg_df = agg_df.reset_index()
agg_df.head()
agg_df.shape


#############################################
# TASK 5: Convert the AGE variable to a categorical variable and add it to agg_df.
#############################################
# Convert the numerical variable Age to a categorical variable.
# Create intervals that you think are convincing.
# For example: '0_18', '19_23', '24_30', '31_40', '41_70'

# Define the breakpoints for AGE:
bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]

# Define the labels corresponding to the breakpoints:
mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]

# Categorize age:
agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins, labels=mylabels)
agg_df.head()

# Check the observations for each category
pd.crosstab(agg_df["AGE"],agg_df["age_cat"])


#############################################
# TASK 6: Define and add level-based new customers to the dataset as variables.
#############################################
# Define a variable named customers_level_based and add it to the dataset.
# Note!
# After the customer_level_based values are generated with list comprehension, they need to be made unique.
# For example, there can be multiple instances of the following: USA_ANDROID_MALE_0_18
# These should be grouped and the mean of the prices should be taken.

# METHOD 1
# Variable names:
agg_df.columns

# How do we access observation values?
for row in agg_df.values:
    print(row)


# We want to concatenate the VALUES of the COUNTRY, SOURCE, SEX, and age_cat variables side by side and combine them with underscores.
# This can be done with list comprehension.
# Let's process the observation values selected above to select the ones we need:

# method 1
[row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]

# method 2
[row["COUNTRY"].upper() + '_' + row["SOURCE"].upper() + '_' + row["SEX"].upper() + '_' + row["age_cat"].upper() for index, row in agg_df.iterrows()]

# method 3
agg_deneme=agg_df.drop(["AGE", "PRICE",'customers_level_based'], axis=1)
agg_deneme.head()
['_'.join(i).upper() for i in agg_deneme.values]
agg_deneme["customers_level_based"] =['_'.join(i).upper() for i in agg_deneme.values]
agg_deneme.head()

# method 4
agg_df[['COUNTRY', 'SOURCE', 'SEX', 'age_cat']].apply(lambda x: '_'.join(x).upper(),axis=1)



# Add to the dataset:
agg_df["customers_level_based"] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'age_cat']].apply(lambda x: '_'.join(x).upper(),axis=1)
agg_df.head()

# Remove unnecessary variables:
agg_df1 = agg_df[["customers_level_based", "PRICE"]]
agg_df1.head()



# We are one step closer to our goal.
# There is a small problem here. There will be many of the same segments.
# For example, there can be multiple instances of the segment USA_ANDROID_MALE_0_18.
# Let's check:
agg_df1["customers_level_based"].value_counts()


# Therefore, after performing groupby by segments, we should take the average prices and make the segments unique.
agg_df1 = agg_df1.groupby("customers_level_based").agg({"PRICE": "mean"})

# customers_level_based is in the index. Let's convert it to a variable.
agg_df1.reset_index(inplace=True)
agg_df1.head()

# Let's check. We expect each persona to have only one instance:
agg_df1["customers_level_based"].value_counts()
agg_df1.head()


#############################################
# TASK 7: Divide new customers (USA_ANDROID_MALE_0_18) into segments.
#############################################
# Divide by segments based on PRICE,
# Add the segments to agg_df with the name "SEGMENT",
# Describe the segments,

agg_df1["SEGMENT"]= pd.qcut(agg_df1["PRICE"], 4, labels=["D", "C", "B", "A"]) #küçükten büyüğe !!!
agg_df1.head(30)


agg_df1.groupby("SEGMENT").agg({"PRICE": ["mean"]}).sort_values('SEGMENT', ascending=False)

#############################################
# TASK 8: Classify new incoming customers and estimate their potential revenue.
#############################################
# What segment does a Turkish woman, aged 33, using an Android device belong to, and what is the expected average revenue?
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df1[agg_df1["customers_level_based"] == new_user]



# What segment does a French woman, aged 35, using an iOS device belong to, and what is the expected average revenue?
new_user2 = "FRA_IOS_FEMALE_31_40"
agg_df1[agg_df1["customers_level_based"] == new_user2]

