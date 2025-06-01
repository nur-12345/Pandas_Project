# Pandas_Project
This project analyzes a dataset of mobile applications from the Google Play Store. It focuses on data cleaning, transformation, and exploratory data analysis (EDA) using pandas, numpy, matplotlib, and seaborn.
 🧰 Technologies Used
Python 🐍
pandas 🧮
numpy 🔢
matplotlib 📊
seaborn 🎨

📂 Dataset
The dataset should be a .csv file with app-related information such as:

App Name
Size
Installs
Price
Rating
Content Rating
Genre
Last Updated
Category
Type (Free/Paid)
Note: Make sure to place your dataset file in the same directory as the script or notebook.
📌 Features & Steps
✅ Data Cleaning:

Removed non-numeric characters from Size, Installs, and Price.
Converted appropriate columns to numeric or datetime.
Handled missing or inconsistent values like "Varies with device".
Simplified genre columns to take only the primary genre.
📊 Exploratory Data Analysis (EDA):

Count and distribution plots for app types, content ratings, and categories.
Boxplots for comparing app ratings based on app type and content rating.
Scatter plots and heatmaps to identify correlations and trends.
Rating distribution visualization with Kernel Density Estimation (KDE).
📸 Visualizations Preview
Sample visualizations include:

📈 Rating Distribution
📦 Boxplot of Ratings vs App Type
🧱 Countplot of Content Ratings
🌡️ Heatmap of Correlations
🎯 Scatter Plot of Price vs Category
🚀 How to Run
Clone the repository:
git clone https://github.com/your-username/app-data-analysis.git
cd app-data-analysis
Install the required packages:
pip install pandas numpy matplotlib seaborn
Open the Jupyter Notebook or run the Python script:
jupyter notebook
Load the dataset and execute the cells in order.
📁 File Structure
Pandas-Project/
│
├── google.py       
├── README.md                  
└── googleplaystore.csv           
🙋‍♀️ Author
Nupur

📜 License
This project is open-source and free to use under the MIT License.

Let me know if you'd like me to generate this as a downloadable file or help with uploading it to your GitHub repo.
