## Dataset

This project uses the **100,000 UK Used Car Data Set** sourced from Kaggle.

### Overview

- **Size:** 100,000 used car listings  
- **Source:** Scraped UK car market data  
- **License:** CC0 (Public Domain)  
- **Update Frequency:** Quarterly  

The dataset contains cleaned and structured information about used cars across multiple manufacturers.

### Features

The dataset includes the following key attributes:

- `price` — vehicle price  
- `transmission` — manual / automatic / semi-auto  
- `mileage` — total miles driven  
- `fuelType` — petrol / diesel / hybrid / electric  
- `tax` — road tax  
- `mpg` — miles per gallon  
- `engineSize` — engine capacity  

Duplicates have been removed and columns cleaned. The dataset is also split by car manufacturer for easier analysis.

### Background

The dataset was originally created to:
- Estimate **optimal resale price** for used cars  
- Analyze **depreciation trends** (age, mileage impact)  
- Compare models (e.g. Ford Focus vs Mercedes C-Class)  

It was later expanded into a **general car price regression dataset**.

### Use Case in This Project

In this project, the dataset is used to:
- Train a **machine learning model** to predict used car prices  
- Perform **feature engineering** (e.g. age, mileage transformations)  
- Analyze **market patterns and pricing behavior**

### Source

👉 [Kaggle Dataset Link](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes)
