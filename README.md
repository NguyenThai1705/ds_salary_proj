## Data Science Salary Estimator: Project Overview
* Create a tool that estimates data science salaries (MSE ~ $ 13k/year) to help data scientists negotiate their income when they get a job.
* Engineered features from the text of each job description to quantity the value companies put on python, r studio, excel, aws, and spark.
* Optimized Linear, Lasso, and Random Forest Regressors using GridsearchCV to get the best model.
* Build a client facing API using Flask.

## Code and Resources Used
**Python Version:** 3.8  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, flask, json, pickle.  
**For Web Framework Requirements:** '''pip install -r requirements.txt'''  
**Glassdoor dataset:** https://github.com/PlayingNumbers/ds_salary_proj  
**Productionization:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2  

## Glassdoor Dataset 
1000 job postings from Glassdoor.com in github repo (above). With each jobs, we got the following:
* Job title
* Salary Estimate
* Job Description
* Rating
* Company
* Location
* Company Headquarters
* Company Size
* Company Founded Date
* Type of Ownership
* Industry
* Sector
* Revenue
* Competitors

## Data Cleaning
After take off the dataset, i needed to clean it up so that it was unable for our model. I made the following changes and created the following variables:
* Parsed numeric data out of salary
* Made columns for employee provided and hourly wages
* Removed rows with out salary
* Parsed rating out of company text
* Made a new column for company state
* Added a column for if the job was at the company's headquarters
* Tranformed company founded date into company's age
* Made columns for if different skills were listed in the job description:
    * Python
    * R
    * Excel
    * AWS
    * Spark
* Column for simplified job title and seniority
* Column for description length 

## EDA 
I look at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot table.

![alt text](https://github.com/NguyenThai1705/ds_salary_proj/blob/main/correlations.png "Correlations")
![alt text](https://github.com/NguyenThai1705/ds_salary_proj/blob/main/job_opportunities_by_states.png "jov opportunities by states")

## Model building

First, I transformed the categorical variables into dummies variables. I also splited data into train and test datasets with a test size of 20%.
I tried three different models and evaluated them using Mean Absolute Error. Each models by following:
* **Multiple Linear Regression** - Baseline for the model
* **Lasso Regression** - Because of the sparse data from many categorical variables
* **Random Forest** - Because the sparsity associated with the data, then I used GridsearchCV to find the best parameters of the model.

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets.
* **Random Forest** : MAE = 13.12
* **Linear Regression** : MAE = 21.55
* **Lasso Regression** : MAE = 20.59

## Productionization
I built a flask API endpoint that was hosted a local webserver. The API endpoint takes in a request with a list of values from a job listing and returns an estimated salary.



