import pandas as pd 
                    
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("ecommerce_customer_data.csv")
print(data.head())
print("\n")
# column info in the dataset
print(data.info())
print("\n")

# printing all the columns in dataset
print(data.columns)
print("\n")

 # Summary statistics for numeric columns
numeric_summary = data.describe()
print(numeric_summary)
print("\n")

# Summary for non-numeric columns
categorical_summary = data.describe(include='object')
print(categorical_summary) 
print("\n")

# look if your data contains any missing values or not:
print(data.isnull().sum()) 
print("\n")

# print the user id, the type of item purchased by the user along with the number of items purchased
data1 = ['User_ID','Device_Type','Total_Purchases']
condition = data["Total_Purchases"] > 0
print(data.loc[condition, data1])
 
# Calculate churn rate -> the annual percentage rate at which customers stop subscribing to a service / employees leave a job.
data['Churned'] = data['Total_Purchases'] == 0
churn_rate = data['Churned'].mean()
print(f"Churn Rate:{churn_rate}")

# Histogram for 'Age'
plt.hist(data['Age'], bins=20, color='skyblue', edgecolor='black') 
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75) 
plt.show()

gender_counts = data['Gender'].value_counts().reset_index()
print(gender_counts)
gender_counts.columns = ['Gender', 'Count'] # rename colums
plt.bar(gender_counts['Gender'], gender_counts['Count']) #data
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# # Grouped Analysis
# Average Total Pages Viewed by Gender
gender_grouped = data.groupby('Gender')['Total_Pages_Viewed'].mean().reset_index()
plt.bar(gender_grouped['Gender'],gender_grouped['Total_Pages_Viewed'],color=['blue', 'orange'])
plt.title('Average Total Pages Viewed by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Total Pages Viewed')
plt.show()

# 'Product_Browsing_Time' vs 'Total_Pages_Viewed' 
plt.scatter(data['Product_Browsing_Time'], data['Total_Pages_Viewed'])
# Linear regression 
m, b = np.polyfit(data['Product_Browsing_Time'], data['Total_Pages_Viewed'], 1)
plt.plot(data['Product_Browsing_Time'], m * data['Product_Browsing_Time'] + b, color='red')
plt.title('Product Browsing Time vs. Total Pages Viewed')
plt.xlabel('Product Browsing Time')
plt.ylabel('Total Pages Viewed')
plt.show()

# Average Total Pages Viewed by Devices
devices_grouped = data.groupby('Device_Type')['Total_Pages_Viewed'].mean().reset_index()
plt.bar(devices_grouped['Device_Type'], devices_grouped['Total_Pages_Viewed'], color=['green', 'red', 'purple'])
plt.title('Average Total Pages Viewed by Devices')
plt.xlabel('Device Type')
plt.ylabel('Average Total Pages Viewed')
plt.show()

# CLV -> Customer Lifetime Value -> Customer Lifetime Value = (Customer Value * Average Customer Lifespan)
data['CLV'] = (data['Total_Purchases'] * data['Total_Pages_Viewed']) / data['Age']
data['Segment'] = pd.cut(data['CLV'], bins=[1, 2.5, 5, float('inf')], labels=['Low Value', 'Medium Value', 'High Value'])
segment_counts = data['Segment'].value_counts().reset_index()
segment_counts.columns = ['Segment', 'Count']
plt.bar(segment_counts['Segment'], segment_counts['Count'], color=['blue', 'orange', 'green'])
plt.title('Customer Segmentation by CLV')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.show()

# DeviceType vs total purchase graph
plt.bar(data['Device_Type'], data['Total_Purchases'], color=['blue'])
plt.title('Total Purchases by Device Type')
plt.xlabel('Device Type')
plt.ylabel('Total Purchases')
plt.show()

# Gender Distribution Pie Chart
gender_distribution = data['Gender'].value_counts()
plt.pie(gender_distribution, labels=gender_distribution.index,autopct='%1.2f%%', startangle=90)
plt.title("Users: Gender Distribution")
plt.show()

# Location Distribution Pie Chart
location_distribution = data['Location'].value_counts()
plt.pie(location_distribution,labels=location_distribution.index,autopct="%1.2f%%", startangle=90)
plt.title("Location based distribution of users")
plt.show()

# heat map
heatmap_data = data.pivot_table(index='Product_Browsing_Time', columns='Total_Pages_Viewed', 
                                values='User_ID', aggfunc='count', fill_value=0)
plt.figure(figsize=(10, 8))
plt.imshow(heatmap_data, cmap='viridis', origin='lower', interpolation='none')
plt.colorbar(label='Count')
plt.xlabel('Total Pages Viewed')
plt.ylabel('Product Browsing Time')
plt.xticks(np.arange(len(heatmap_data.columns)), heatmap_data.columns)
plt.yticks(np.arange(len(heatmap_data.index)), heatmap_data.index)
plt.title('Heatmap of Product Browsing Time vs Total Pages Viewed')
plt.show()
