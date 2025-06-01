# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the dataset
df = pd.read_csv("/Users/nupurshivani/Downloads/Pandas-Project/googleplaystore.csv")

# Display the first 5 rows
print(df.head())

# # Check the shape (rows, columns)
print(df.shape)

# # Display column names
print(df.columns)

# Replace spaces in column names with underscores for easy access
df.columns = df.columns.str.replace(" ", "_")
print(df.columns)

# Check shape again
print(df.shape)

# Check data types
print(df.dtypes)

# Handling missing values
print(df.isnull().sum())  # Print missing values count per column

# Fill missing values in 'Rating' column with the median
rating_median = df["Rating"].median()
print("Rating Median:", rating_median)
df["Rating"].fillna(rating_median, inplace=True)

# Drop remaining rows with any missing values
df.dropna(inplace=True)
print("Remaining Missing Values:", df.isnull().sum().sum())

# Dataset overview
print(df.info())

# Describe 'Reviews' column before converting to integer
print(df["Reviews"].describe())

# Convert 'Reviews' column to numeric
df["Reviews"] = df["Reviews"].astype("int64")
print(df["Reviews"].describe().round())

# Handling 'Size' column
print("Unique Sizes:", len(df["Size"].unique()))
print(df["Size"].unique())  # View unique size values

# Remove 'M' and 'k' suffixes from 'Size' column
df["Size"].replace("M", "", regex=True, inplace=True)
df["Size"].replace("k", "", regex=True, inplace=True)

print("After cleaning suffixes:", df["Size"].unique())

# Replace 'Varies with device' with median of other values
size_median = df[df["Size"] != "Varies with device"]["Size"].astype(float).median()
df["Size"].replace("Varies with device", size_median, inplace=True)

# Convert 'Size' column to numeric
df["Size"] = pd.to_numeric(df["Size"])
print(df["Size"].head())
print(df["Size"].describe().round())

# Handling 'Installs' column
print(df["Installs"].unique())

# Remove '+' and ',' characters from 'Installs' column
df["Installs"] = df["Installs"].apply(lambda x: x.replace("+", "").replace(",", ""))
df["Installs"] = df["Installs"].astype(int)
print(df["Installs"].unique())

# Handling 'Price' column
print(df["Price"].unique())

# Remove '$' and convert to float
df["Price"] = df["Price"].apply(lambda x: x.replace("$", ""))
df["Price"] = df["Price"].astype(float)
print(df["Price"].unique())

# Check number of unique genres
print("Unique Genres:", len(df["Genres"].unique()))

# Show some sample genres
print(df["Genres"].head(10))

# Split multiple genres and keep only the first one
df["Genres"] = df["Genres"].str.split(";").str[0]
print("After cleaning genres:", len(df["Genres"].unique()))
print(df["Genres"].unique())

# Replace a specific genre name for simplification
df["Genres"].replace("Music & Audio", "Music", inplace=True)

# Handling 'Last Updated' column
print(df["Last_Updated"].head())

# Convert to datetime format
df["Last_Updated"] = pd.to_datetime(df["Last_Updated"])

# Check head and data types again
print(df.head())
print(df.dtypes)

# Visualization: Distribution of app types (Free vs Paid)
df["Type"].value_counts().plot(kind="bar", color="red")
plt.title("Free & Paid")
plt.show()

# Boxplot of Ratings based on app type
sns.boxplot(x="Type", y="Rating", data=df)
plt.title("Rating by App Type")
plt.show()

# Count of content ratings
sns.countplot(y="Content_Rating", data=df)
plt.title("Content Rating Counts")
plt.show()

# Boxplot of rating by content rating
sns.boxplot(x="Content_Rating", y="Rating", data=df)
plt.title("Content Rating vs Rating", size=20)
plt.show()

# Barplot of app counts per category
cat_num = df["Category"].value_counts()
sns.barplot(x=cat_num.values, y=cat_num.index)
plt.title("App Counts per Category", size=20)
plt.show()

# Scatter plot of price vs category
sns.scatterplot(data=df, y="Category", x="Price")
plt.title("Category vs Price", size=20)
plt.show()

# Heatmap showing correlation among numerical features
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, linewidths=0.5, fmt=".2f")

plt.title("Correlation Heatmap", size=20)
plt.show()

# Histogram with KDE of rating column
sns.histplot(df["Rating"], kde=True)
plt.title("Rating Distribution with KDE", size=20)
plt.show()
