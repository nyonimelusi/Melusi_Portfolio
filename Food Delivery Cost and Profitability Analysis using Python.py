#!/usr/bin/env python
# coding: utf-8

# Food Delivery Cost and Profitability Analysis is a comprehensive evaluation aimed at understanding and optimizing the financial dynamics of a food delivery operation. The goal is to identify areas where the service can reduce costs, increase revenue, and implement pricing or commission strategies that enhance profitability. If you're interested in learning how to perform a cost and profitability analysis of a business operation, this article is for you. Here, I'll guide you through the process of conducting a Food Delivery Cost and Profitability Analysis using Python.

# # Food Delivery Cost and Profitability Analysis: Process We Can Follow

# Food Delivery Cost and Profitability Analysis involves examining all the costs associated with delivering food orders, from direct expenses like delivery fees and packaging to indirect expenses like discounts offered to customers and commission fees paid by restaurants. By juxtaposing these costs against the revenue generated (primarily through order values and commission fees), the analysis aims to provide insights into how profitable the food delivery service is on a per-order basis

# # Below is the process we can follow for the task of Food Delivery Cost and Profitability Analysis:
1.Start by gathering comprehensive data related to all aspects of food delivery operations.
2.Clean the dataset for inconsistencies, missing values, or irrelevant information.
3.Extract relevant features that could impact cost and profitability.
4.Break down the costs associated with each order, including fixed costs (like packaging) and variable costs (like delivery fees and discounts).
5.Determine the revenue generated from each order, focusing on commission fees and the order value before discounts.
6.For each order, calculate the profit by subtracting the total costs from the revenue. Analyze the distribution of profitability across all orders to identify trends.
7.Based on the cost and profitability analysis, develop strategic recommendations aimed at enhancing profitability.
8.Use the data to simulate the financial impact of proposed changes, such as adjusting discount or commission rates.
# # Food Delivery Cost and Profitability Analysis using Python

# In[2]:


import pandas as pd

food_orders = pd.read_csv("food_orders_new_delhi.csv")
print(food_orders.head())


# In[3]:


print(food_orders.info())


# Next Step is Data Cleaning.The dataset contains 1000 entries, and  12 columns. No missing entries.
# Data Preparation Steps
# 
# 1. Convert Date and Time Columns**: Convert the “Order Date and Time” and “Delivery Date and Time” fields to a proper datetime format for analysis.
# 2. Standardize Discounts**: Ensure consistency in the “Discounts and Offers” field by converting it to numeric values or calculating discount amounts if applicable.
# 3. Format Monetary Values**: Ensure all monetary fields are in a consistent, suitable format for calculations and analysis.

# In[4]:


from datetime import datetime

# convert date and time columns to datetime
food_orders['Order Date and Time'] = pd.to_datetime(food_orders['Order Date and Time'])
food_orders['Delivery Date and Time'] = pd.to_datetime(food_orders['Delivery Date and Time'])

# first, let's create a function to extract numeric values from the 'Discounts and Offers' string
def extract_discount(discount_str):
    if 'off' in discount_str:
        # Fixed amount off
        return float(discount_str.split(' ')[0])
    elif '%' in discount_str:
        # Percentage off
        return float(discount_str.split('%')[0])
    else:
        # No discount
        return 0.0

# apply the function to create a new 'Discount Value' column
food_orders['Discount Percentage'] = food_orders['Discounts and Offers'].apply(lambda x: extract_discount(x))

# for percentage discounts, calculate the discount amount based on the order value
food_orders['Discount Amount'] = food_orders.apply(lambda x: (x['Order Value'] * x['Discount Percentage'] / 100)
                                                   if x['Discount Percentage'] > 1
                                                   else x['Discount Percentage'], axis=1)

# adjust 'Discount Amount' for fixed discounts directly specified in the 'Discounts and Offers' column
food_orders['Discount Amount'] = food_orders.apply(lambda x: x['Discount Amount'] if x['Discount Percentage'] <= 1
                                                   else x['Order Value'] * x['Discount Percentage'] / 100, axis=1)

print(food_orders[['Order Value', 'Discounts and Offers', 'Discount Percentage', 'Discount Amount']].head(), food_orders.dtypes)


# The data has been prepared with the following adjustments:
# 
# 1. **Date and Time Conversion**: The "Order Date and Time" and "Delivery Date and Time" columns have been successfully converted to datetime format.
# 2. **Discount Amount Calculation**: A new column, "Discount Amount," has been added, which calculates discounts by extracting percentage or fixed amounts from the "Discounts and Offers" column and applying them to the order value.
# 3. **Discount Percentage Added**: The "Discount Percentage" column has been introduced to represent the rate or fixed amount discount directly for each order.

#  COST AND PROFITABILITY ANALYSIS
# 
# To conduct the cost analysis, we'll account for the following expenses per order:
# 
# 1. **Delivery Fee**: The cost incurred for delivering the order.
# 2. **Payment Processing Fee**: The fee for handling payment transactions.
# 3. **Discount Amount**: The discount provided on each order.
# 
# We'll calculate the **Total Cost** by summing these expenses for each order. To determine profitability, we'll calculate **Net Profit** by subtracting total costs (including discounts) from the platform's **Commission Fee** revenue. This will provide insights into the platform's overall cost structure and profitability.

# In[5]:


# calculate total costs and revenue per order
food_orders['Total Costs'] = food_orders['Delivery Fee'] + food_orders['Payment Processing Fee'] + food_orders['Discount Amount']
food_orders['Revenue'] = food_orders['Commission Fee']
food_orders['Profit'] = food_orders['Revenue'] - food_orders['Total Costs']

# aggregate data to get overall metrics
total_orders = food_orders.shape[0]
total_revenue = food_orders['Revenue'].sum()
total_costs = food_orders['Total Costs'].sum()
total_profit = food_orders['Profit'].sum()

overall_metrics = {
    "Total Orders": total_orders,
    "Total Revenue": total_revenue,
    "Total Costs": total_costs,
    "Total Profit": total_profit
}

print(overall_metrics)


# # Summary of Food Delivery Operations Metrics
# 
# - **Total Orders**: 1,000
# - **Total Revenue (from Commission Fees)**: 126,990 INR
# - **Total Costs**: 232,709.85 INR (including delivery fees, payment processing fees, and discounts)
# - **Total Profit**: -105,719.85 INR
# 
# The analysis shows a **net loss** as costs exceed revenue, indicating potential unsustainability in the current commission, delivery fees, and discount strategies.
# 
# ### Visualization Plan:
# 1. **Histogram**: To show the distribution of profits per order.
# 2. **Pie Chart**: To represent the proportion of costs.
# 3. **Bar Chart**: To compare revenue, costs, and profit.
# 
# Let's plot the histogram first.

# In[6]:


import matplotlib.pyplot as plt

# histogram of profits per order
plt.figure(figsize=(10, 6))
plt.hist(food_orders['Profit'], bins=50, color='skyblue', edgecolor='black')
plt.title('Profit Distribution per Order in Food Delivery')
plt.xlabel('Profit')
plt.ylabel('Number of Orders')
plt.axvline(food_orders['Profit'].mean(), color='red', linestyle='dashed', linewidth=1)
plt.show()


# The histogram reveals a broad distribution of profit per order, with a significant number of orders showing a loss (profits below zero). The red dashed line marks the average profit, which falls in the negative range, emphasizing the overall loss-making scenario.
# 
# Next, let's examine the proportion of total costs:

# In[7]:


# pie chart for the proportion of total costs
costs_breakdown = food_orders[['Delivery Fee', 'Payment Processing Fee', 'Discount Amount']].sum()
plt.figure(figsize=(7, 7))
plt.pie(costs_breakdown, labels=costs_breakdown.index, autopct='%1.1f%%', startangle=140, colors=['tomato', 'gold', 'lightblue'])
plt.title('Proportion of Total Costs in Food Delivery')
plt.show()


# The pie chart shows the distribution of total costs across delivery fees, payment processing fees, and discount amounts. Discounts make up a substantial portion, indicating that promotional strategies may be having a significant effect on overall profitability.
# 
# Next, let’s compare total revenue, total costs, and total profit (or net loss, in this case):

# In[8]:


# bar chart for total revenue, costs, and profit
totals = ['Total Revenue', 'Total Costs', 'Total Profit']
values = [total_revenue, total_costs, total_profit]

plt.figure(figsize=(8, 6))
plt.bar(totals, values, color=['green', 'red', 'blue'])
plt.title('Total Revenue, Costs, and Profit')
plt.ylabel('Amount (INR)')
plt.show()


# The bar chart compares total revenue, total costs, and total profit. It highlights the disparity between revenue and costs, clearly illustrating that costs exceed revenue, resulting in a net loss.

# # A New Approach to Achieving Profitability

# From the analysis so far, we’ve identified that excessive discounts on food orders are driving significant losses. To turn things around, we need to develop a new strategy that finds the optimal balance between offering discounts and charging commissions. By closely examining the characteristics of profitable orders, we can aim to:
# 
# - Determine a new average commission percentage based on profitable orders.
# - Establish a new average discount percentage that still allows for profitability, serving as a guideline for future discounts.
# 
# Using these new averages, we can propose adjustments that not only make individual orders profitable but also improve profitability across the board. Let’s proceed by calculating:
# 
# - The average commission percentage for profitable orders.
# - The average discount percentage for profitable orders.

# In[9]:


# filter the dataset for profitable orders
profitable_orders = food_orders[food_orders['Profit'] > 0]

# calculate the average commission percentage for profitable orders
profitable_orders['Commission Percentage'] = (profitable_orders['Commission Fee'] / profitable_orders['Order Value']) * 100

# calculate the average discount percentage for profitable orders
profitable_orders['Effective Discount Percentage'] = (profitable_orders['Discount Amount'] / profitable_orders['Order Value']) * 100

# calculate the new averages
new_avg_commission_percentage = profitable_orders['Commission Percentage'].mean()
new_avg_discount_percentage = profitable_orders['Effective Discount Percentage'].mean()

print(new_avg_commission_percentage, new_avg_discount_percentage)


# Based on our analysis of profitable orders, we’ve identified new average values that could represent a “sweet spot” for commission and discount percentages:
# 
# - **New Average Commission Percentage:** 30.51%
# - **New Average Discount Percentage:** 5.87%
# 
# The commission percentage for profitable orders is significantly higher than the overall average, suggesting that increasing the commission rate could be key to achieving profitability. Meanwhile, the discount percentage for profitable orders is notably lower, indicating that offering smaller discounts may help maintain profitability without significantly reducing order volume.
# 
# A strategy that targets a commission rate around 30% and a discount rate near 6% could potentially enhance profitability across all orders.
# 
# To visualize this, we can compare the impact of actual versus recommended discounts and commissions. For this, we need to:
# 
# 1. **Calculate profitability per order** using the actual discounts and commissions in the dataset.
# 2. **Simulate profitability per order** using the recommended discount rate (6%) and commission rate (30%) to assess the potential impact on profitability.
# 
# This comparison will provide a clear visual representation of how adopting the recommended discount and commission rates could improve overall profitability. Here's how to visualize this comparison:

# In[13]:


get_ipython().system('pip install seaborn')
import seaborn as sns


# In[14]:


# simulate profitability with recommended discounts and commissions
recommended_commission_percentage = 30.0  # 30%
recommended_discount_percentage = 6.0    # 6%

# calculate the simulated commission fee and discount amount using recommended percentages
food_orders['Simulated Commission Fee'] = food_orders['Order Value'] * (recommended_commission_percentage / 100)
food_orders['Simulated Discount Amount'] = food_orders['Order Value'] * (recommended_discount_percentage / 100)

# recalculate total costs and profit with simulated values
food_orders['Simulated Total Costs'] = (food_orders['Delivery Fee'] +
                                        food_orders['Payment Processing Fee'] +
                                        food_orders['Simulated Discount Amount'])

food_orders['Simulated Profit'] = (food_orders['Simulated Commission Fee'] -
                                   food_orders['Simulated Total Costs'])

# visualizing the comparison
import seaborn as sns

plt.figure(figsize=(14, 7))

# actual profitability
sns.kdeplot(food_orders['Profit'], label='Actual Profitability', fill=True, alpha=0.5, linewidth=2)

# simulated profitability
sns.kdeplot(food_orders['Simulated Profit'], label='Estimated Profitability with Recommended Rates', fill=True, alpha=0.5, linewidth=2)

plt.title('Comparison of Profitability in Food Delivery: Actual vs. Recommended Discounts and Commissions')
plt.xlabel('Profit')
plt.ylabel('Density')
plt.legend(loc='upper left')
plt.show()


# This analysis compares the profitability of a food delivery company under two scenarios: the current scenario using actual discounts and commissions, and a simulated scenario using recommended discounts (6%) and commissions (30%). The goal is to assess the potential impact of these adjustments on the company's overall profitability.
# 
# Key Findings
# 
# Current Scenario: The distribution of profitability per order in the current scenario is relatively wide, with a significant number of orders resulting in losses. This suggests that the company's pricing strategy and cost structure may be impacting its profitability.
# Simulated Scenario: The recommended adjustments to discounts and commissions lead to a more favorable distribution of profitability per order. The data indicates a higher proportion of profitable orders and a shift towards higher profit levels.
# Conclusion
# 
# Based on this analysis, implementing the recommended discounts and commissions could have a positive impact on the food delivery company's profitability. By reducing discounts and increasing commission fees, the company may be able to improve its revenue per order and reduce the number of unprofitable orders. However, further analysis is needed to evaluate the potential impact of these changes on customer satisfaction and market share.

# In[ ]:




