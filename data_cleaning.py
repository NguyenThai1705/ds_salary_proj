# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 10:05:33 2021

@author: nguye
"""

import pandas as pd 

df = pd.read_csv('glassdoor_jobs.csv')

#salary parsing
df = df[df['Salary Estimate'] != '-1']
df['per_hour'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary' in x.lower() else 0)

salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
min_char = salary.apply(lambda x: x.replace('$', '').replace('K', ''))
min_hr = min_char.apply(lambda x: x.lower().replace('per hour', '').replace('employer provided salary:', ''))

df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2

#company text only
df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'].lower()[:-3], axis = 1)

#state field
df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])
df.job_state.value_counts()

df['same_location'] = df.apply(lambda x: 1 if x['job_state'] in x['Headquarters'] else 0, axis = 1)
#age of company
df['age'] = df.Founded.apply(lambda x: x if x < 1 else 2021 - int(x))

#parsing of job description
#python
df['python'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)

#r studio
df['r_studio'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() else 0)

#spark
df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)

#aws
df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)

#excel
df['excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)


df_out = df.drop(['Unnamed: 0'], axis = 1)
df_out.to_csv('salary_data_cleaned.csv', index = False)
