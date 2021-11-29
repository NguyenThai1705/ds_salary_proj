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

## EDA variable
I look at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot table.


