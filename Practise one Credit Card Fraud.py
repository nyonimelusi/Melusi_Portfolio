#!/usr/bin/env python
# coding: utf-8

# In[3]:


legittransaction = 680
fraudtransaction = 27
totaltransaction = legittransaction + fraudtransaction

legit_percentage = (legittransaction / totaltransaction) * 100
fraud_percentage = (fraudtransaction / totaltransaction) * 100

print("Legit Percentage:", legit_percentage)
print("Fraud Percentage:", fraud_percentage)


# In[5]:


import matplotlib.pyplot as plt

legittransaction = 680
fraudtransaction = 27

labels = ["Legitimate Transactions", "Fraudulent Transactions"]
sizes = [legittransaction, fraudtransaction]

plt.pie(sizes, labels=labels, shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Legitimate Transactions vs Fraudulent Transactions")
plt.show()


# In[1]:


import pandas as pd
df=pd.read_csv("card_transdata.csv",sep=",")
df 


# In[26]:


df.head()


# In[27]:


df.tail()


# In[28]:


df.info()


# In[29]:


df.shape


# In[30]:


df.describe()


# In[31]:


df.columns


# In[32]:


df.dtypes


# CLEANING DATASET BY REMOVING MISSING VALUES AND DUPLICATES

# In[33]:


missing_values = df.isnull().any(axis=1)
print("Rows with missing values:")
print(missing_values)


# In[34]:


duplicate_rows = df[df.duplicated()]
print("Duplicated Rows:")
print(duplicate_rows)


# In[35]:


df.dropna(axis=0, inplace=True)


# In[36]:


df.drop_duplicates(inplace=True)


# In[37]:


chippindf = df[["used_chip","used_pin_number","fraud"]]


# In[48]:


total_transactions = len(chippindf)
total_fraud =chippindf["fraud"].sum()
fraud_by_chip = chippindf[chippindf["used_chip"]==1]["fraud"].sum()
fraud_by_pin = chippindf[chippindf["used_pin_number"] == 1]["fraud"].sum()     


# In[49]:


print("Total transactions:", total_transactions)
print("Total fraud cases:", total_fraud)
print("Fraud cases using chip: {} out of {}".format(fraud_by_chip,total_transactions))
print("Fraud cases using pin: {} out of {}".format(fraud_by_pin,total_transactions))


# In[52]:


import matplotlib.pyplot as plt

labels_chip = ["Non-Fraud","Fraud"]
sizes_chip = [total_transactions - fraud_by_chip,fraud_by_chip]
colors_chip = ["lightskyblue", "lightcoral"]
labels_pin = ["Non-Fraud","Fraud"]
sizes_pin = [total_transactions - fraud_by_pin,fraud_by_pin]
colors_pin = ["lightskyblue", "lightcoral"]
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.pie(sizes_chip,labels=labels_chip, colors=colors_chip, startangle=140)
plt.axis("equal")
plt.title("Chip Transactions")
plt.subplot(1,2,2)
plt.pie(sizes_pin,labels=labels_pin, colors=colors_pin, startangle=140)
plt.axis("equal")
plt.title("Pin Transactions")
plt.suptitle("Fraud cases in Chip and pin transaction")
plt.show()


# The above pie charts representation:

# The two charts represents fraud by : Chip transactions and pin transactions.
# The first pie chart represents fraudlent transactions by chip fraud. The blue part represnts legitimate transactions and the light coral represents percentage of fraud which is a small proportion.
# While on the pin transaction chart the proportion of fraud cases is very small around 273 cases out of a million.
# 

# # ANALYZING REPEAT RETAILER FRAUD PATTERNS

# In[65]:


repeat_retailer_df = df[df["repeat_retailer"]==1]


# In[64]:


fraud_sequences = []
current_sequence = []

for index, row in repeat_retailer_df.iterrows():
    repeat_retailer, is_fraud = row['repeat_retailer'], row['fraud']

    if is_fraud == 1:
        if current_sequence:
            fraud_sequences.append(current_sequence.copy())
        current_sequence = []
    else:
        current_sequence.append('Repeat Retailer' if repeat_retailer == 1 else 'No Repeat Retailer')

for i, sequence in enumerate(fraud_sequences[:10], start=1):
    print(f"Fraud Sequence {i}: {', '.join(sequence)}")



# # Finding Correlation Between Transaction Amount And Fraud

# In[67]:


df.head()


# In[69]:


correlation_df = df[["ratio_to_median_purchase_price","fraud"]]


# In[70]:


correlation = correlation_df["ratio_to_median_purchase_price"].corr(correlation_df["fraud"])
print(f"Correlation between transaction amount and fraud:{correlation}")


# In[72]:


avgnonfraudtransactions = correlation_df[correlation_df["fraud"]==0]["ratio_to_median_purchase_price"].mean()
avgfraudtransactions = correlation_df[correlation_df["fraud"]==1]["ratio_to_median_purchase_price"].mean()
print(f"Average ration to median purchase price for non fraudlent transactions:{avgnonfraudtransactions}")
print(f"Average ration to median purchase price for fraudlent transactions:{avgfraudtransactions}")


# In[77]:


import matplotlib.pyplot as plt

avgfraudtransaction = correlation_df[correlation_df["fraud"] == 1]["ratio_to_median_purchase_price"].mean()

categories = ["Non-fraudulent", "Fraudulent"]
average_ratio = [avgnonfraudtransactions, avgfraudtransaction]

plt.bar(categories, average_ratio, color=['blue', 'red'])
plt.title("Ratio to Median Purchase Price")
plt.show()


# # Analyzing Fraud Cases In Online Transactions

# In[78]:


df.head()


# In[80]:


online_order_df = df[["online_order","fraud"]]


# In[81]:


total_online_orders = online_order_df["online_order"].sum()
total_online_fraud = online_order_df[(online_order_df["fraud"]==1)&(online_order_df["online_order"]==1)]["fraud"].count()
fraud_rate_online = total_online_fraud/total_online_orders
total_offline_orders = len(online_order_df) - total_online_orders
total_offline_fraud = online_order_df[(online_order_df["fraud"]==1)&(online_order_df["online_order"]==0)]["fraud"].count()
fraud_rate_offline = total_offline_fraud/total_offline_orders
print(f"Fraud rate for online transactions: {fraud_rate_online:.2%} ({total_online_fraud} cases out of {total_online_orders} online transactions)")
print(f"Fraud rate for offline transactions: {fraud_rate_offline:.2%} ({total_offline_fraud} cases out of {total_offline_orders} offline transactions)")


# # Conducting Feature Selections With Random Forest

# In[84]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the data (Replace 'your_data.csv' with the actual file name)
data = pd.read_csv('card_transdata.csv')


# In[85]:


df.head()


# In[93]:


X = data.drop("fraud",axis=1)
y = data["fraud"]


# In[103]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Assuming you have already loaded X and y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Fit the model on the training data
rf_classifier.fit(X_train, y_train)

# Get feature importances
feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)

# Print ranked feature importance
print("Ranked Feature Importance:")
print(feature_importances)


# # Building Credit Card Fraud Detection With Machine Learning

# In[105]:


df.head()


# In[22]:


print(df)


# In[23]:


data = pd.read_csv("card_transdata.csv")


# In[24]:


data.head()


# In[13]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[15]:


data = pd.read_csv("card_transdata.csv")

data.head()


# In[16]:


X = data.drop("fraud",axis=1)
y = data["fraud"]


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train,y_train)
new_transaction_features = data.sample(1).drop('fraud',axis=1)
print("\nRandomly sampled features for new transaction:")
print(new_transaction_features)
prediction = rf_classifier.predict(new_transaction_features)
print("\nPrediction for new transaction:")
print("Fraud" if prediction[0] == 1 else "Legitimate")


# In[19]:


new_transaction_features1 = pd.DataFrame({
    'distance_from_home': [85],
    'distance_from_last_transaction': [75],
    'ratio_to_median_purchase_price': [5.1],
    'repeat_retailer': [0],
    'used_chip': [1],
    'used_pin_number': [0],
    'online_order': [0]
})
prediction = rf_classifier.predict(new_transaction_features1)
print("\nPrediction for new transaction:")
print("Fraud" if prediction[0] == 1 else "Legitimate")


# Building Credit Card Detection Model With Logistic Regression

# In[20]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# In[21]:


data.head()


# In[22]:


X = data.drop("fraud",axis=1)
y = data["fraud"]


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
logreg_classifier = LogisticRegression(max_iter=1000, random_state=42)
logreg_classifier.fit(X_train_scaled,y_train)
new_transaction_features1 = pd.DataFrame({
    'distance_from_home': [89],
    'distance_from_last_transaction': [15],
    'ratio_to_median_purchase_price': [2.3],
    'repeat_retailer': [1],
    'used_chip': [0],
    'used_pin_number': [1],
    'online_order': [1]
})
prediction = logreg_classifier.predict(scaler.transform(new_transaction_features1))
print("\nPrediction for New Transaction:")
print("Fraud" if prediction[0] == 1 else "Legitimate")


# Building Credit Card Fraud Modle with SVM (Support Vector Machine)

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load your data
data = pd.read_csv("card_transdata.csv")  # Replace "your_data.csv" with your actual file path

# Assuming 'data' is your DataFrame
X_rf = data.drop("fraud", axis=1)
y_rf = data["fraud"]

# Split the data for RandomForestClassifier
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_rf, y_train_rf)

# Now, for SVM
X_svm = data.drop("fraud", axis=1)
y_svm = data["fraud"]

# Use StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_svm)

# Initialize the SVM classifier
svm_classifier = SVC(kernel="linear", probability=True, random_state=42)
svm_classifier.fit(X_scaled, y_svm)

# Now you can proceed with making predictions or using the SVM classifier as needed





# In[ ]:


data.head()


# In[ ]:


data = pd.read_csv("card_transdata.csv").sample(1000,random_state=42)


# In[ ]:


X = data.drop("fraud",axis=1)
y = data["fraud"]


# In[ ]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
svm_classifier = SVC(kernel="linear", probability=True, random_state=42)
calibrated_svm = CalibratedClassifierCV(svm_classifier)
calibrated_svm.fit(X_scaled, y)

distance_from_home = float(input("Enter Distance From Home: "))
distance_from_last_transaction = float(input("Enter Distance From Last Transaction: "))
ratio_to_median_purchase_price = float(input("Enter Ratio to Median Purchase Price: "))
repeat_retailer = int(input("Enter Repeat Retailer (0 or 1): "))
used_chip = int(input("Enter Used Chip (0 or 1): "))
used_pin_number = int(input("Enter Used Pin Number (0 or 1): "))
online_order = int(input("Enter Online Order (0 or 1): "))

new_transaction_features = pd.DataFrame({
    'distance_from_home': [distance_from_home],
    'distance_from_last_transaction': [distance_from_last_transaction],
    'ratio_to_median_purchase_price': [ratio_to_median_purchase_price],
    'repeat_retailer': [repeat_retailer],
    'used_chip': [used_chip],
    'used_pin_number': [used_pin_number],
    'online_order': [online_order]
})

scaled_transaction = scaler.transform(new_transaction_features)
prediction = calibrated_svm.predict(scaled_transaction)
probability_of_fraud = calibrated_svm.predict_proba(scaled_transaction)[:, 1][0]

print("\nPrediction for New Transaction:")
print("Fraud" if prediction[0] == 1 else "Legitimate")
print(f"Probability of Fraud: {probability_of_fraud * 100:.2f}%")


# In[ ]:





# Evaluating Model Performance With Precision Recall, And F1 Score 

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score


# In[ ]:


data = pd.read_csv("card_transdata.csv")


# In[ ]:


data.head()


# In[ ]:


X = data.drop("fraud",axis=1)
y = data["fraud"]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
logreg_classifier = LogisticRegression(max_iter=1000, random_state=42)
logreg_classifier.fit(X_train_scaled,y_train)
new_transaction_features1 = pd.DataFrame({
    'distance_from_home': [89],
    'distance_from_last_transaction': [15],
    'ratio_to_median_purchase_price': [2.3],
    'repeat_retailer': [1],
    'used_chip': [0],
    'used_pin_number': [1],
    'online_order': [1]
})
prediction = logreg_classifier.predict(scaler.transform(new_transaction_features1))
print("\nPrediction for New Transaction:")
print("Fraud" if prediction[0] == 1 else "Legitimate")

y_pred = logreg_classifier.predict(X_test_scaled)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("\nEvaluation Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"f1 score: {f1:.4f}")
print(f"accuracy: {accuracy:.4f}")


# In[ ]:




