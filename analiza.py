
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r'data\cleaned_data.csv')

df = pd.DataFrame(data)
df = df.loc[:,df.columns !='Unnamed: 0']
df = df.loc[:,df.columns !='NumberOfPeopleLiableToMaintenance']
df = df.loc[:,df.columns !='Decision']



info = data.info()
shape=data.shape

desc_stats = data.describe()

missing_values = data.isna().sum()

#print(info,shape,desc_stats,missing_values)

# Z podanych informacji możemy wywnioskować, że zbiór danych zawiera 9 kolumn liczbowych
# i 13 kolumn kategorycznych. W zbiorze danych nie ma brakujących wartosci.

# Kredytobiorcy brali pożyczki na srednio 20.9 miesiaca, srednia kwota kredytu wynosila 3271 DM,
# a najwieksza wartosc kredytu wyniosla 18424 DM.


correlation_matrix = df.corr(numeric_only=True)

# Plotting the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Macierz korelacji między zmiennymi numerycznymi')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['CreditAmount'], bins=30, kde=True)
plt.title('Rozkład kwoty kredytu')
plt.xlabel('Kwota kredytu')
plt.ylabel('Liczba klientów')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=30, kde=True)
plt.title('Rozkład wieku klientów')
plt.xlabel('Wiek')
plt.ylabel('Liczba klientów')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Duration'], bins=30, kde=True)
plt.title('Rozkład czasu trwania kredytu')
plt.xlabel('Czas trwania kredytu (w miesiącach)')
plt.ylabel('Liczba klientów')
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.countplot(data, y= 'CreditHistory', order=df['CreditHistory'].value_counts().index)
for container in ax.containers:
    ax.bar_label(container)
plt.title('')
plt.xlabel('Liczba klientów')
plt.ylabel('Status płatności poprzednich kredytów')
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.countplot(data, y= 'Purpose', order=df['Purpose'].value_counts().index)
for container in ax.containers:
    ax.bar_label(container)
plt.title('')
plt.xlabel('Liczba klientów')
plt.ylabel('Cel kredytu')
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.countplot(data, y= 'PersonalStatusAndSex', order=df['PersonalStatusAndSex'].value_counts().index)
for container in ax.containers:
    ax.bar_label(container)
plt.title('')
plt.xlabel('Liczba klientów')
plt.ylabel('Płeć i status związku')
plt.show()


