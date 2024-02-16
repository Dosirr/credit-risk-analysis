# importing needed packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# importing data
data = pd.read_csv(r'data\cleaned_data.csv')

# changing data to dataframe, getting rid of unneeded columns
df = pd.DataFrame(data)
df = df.loc[:,df.columns !='Unnamed: 0']
df = df.loc[:,df.columns !='NumberOfPeopleLiableToMaintenance']
df = df.loc[:,df.columns !='Decision']
# checking for NAs in data
missing_values = data.isna().sum()
# descriptive statistics
desc_stats = data.describe()
# from the data we can draw the following conclusions: 
#      Loans were taken for an average of 21 months
#      The average loan amount was 3271 DM, the highest loan was 18424 DM
#      The average age of clients is 35 years old

# Plotting the correlation matrix
correlation_matrix = df.corr(numeric_only=True)

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation matrix')
plt.show()

# from the correlation matrix we can see that there is moderate positive correlation between credit amount and duration of credit, this tells us that the bigger credit is, the longer it takes to pay it off
# We can also spot weak positive correlation between age and residence duration and weak negative correlation between installment rate in percent and credit amount

plt.figure(figsize=(10, 6))
sns.histplot(data['CreditAmount'], bins=30, kde=True)
plt.title('Distribution of credit amount')
plt.xlabel('Credit amount')
plt.ylabel('Customers')
plt.show()

# We can see at this chart that most of the credits had amount less than 5000 DM. This distribution is definitely right-handed

plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=30, kde=True)
plt.title('Distribution of age')
plt.xlabel('Age')
plt.ylabel('Customers')
plt.show()

# We can see that distribution of this chart is also right-handed. Most of the clients was in the range of 20-40 years old

plt.figure(figsize=(10, 6))
sns.histplot(data['Duration'], bins=30, kde=True)
plt.title('Distribution of credit duration')
plt.xlabel('Credit duration (months)')
plt.ylabel('Customers')
plt.show()

# We can see the seasonality of the credit duration. The credits was taken mostly for 12 months(1 year), 18 months (1.5 year), 24 months(2 years) and 36 months(3 years).

plt.figure(figsize=(10, 6))
ax = sns.countplot(data, y= 'CreditHistory', order=df['CreditHistory'].value_counts().index)
for container in ax.containers:
    ax.bar_label(container)
plt.title('')
plt.xlabel('Customers')
plt.ylabel('Credit history')
plt.show()

# Most of the customers paid their previous credits duly

plt.figure(figsize=(10, 6))
ax = sns.countplot(data, y= 'Purpose', order=df['Purpose'].value_counts().index)
for container in ax.containers:
    ax.bar_label(container)
plt.title('')
plt.xlabel('Customers')
plt.ylabel('Purpose')
plt.show()

# The most common reason for loan was radio/TV 

plt.figure(figsize=(10, 6))
ax = sns.countplot(data, y= 'PersonalStatusAndSex', order=df['PersonalStatusAndSex'].value_counts().index)
for container in ax.containers:
    ax.bar_label(container)
plt.title('')
plt.xlabel('Customers')
plt.ylabel('Sex and mariage status')
plt.show()
# Most often, men who were not in a formal relationship applied for credit.

