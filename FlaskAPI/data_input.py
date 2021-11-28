# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 10:56:51 2021

@author: nguye
"""
import pandas as pd

columns_model = ['Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue',
       'comp_count', 'per_hour', 'employer_provided', 'age', 'python',
       'r_studio', 'spark', 'aws', 'excel', 'title_simp', 'seniority']

columns_dummies = ['Rating', 'comp_count', 'per_hour', 'employer_provided', 'age', 
                   'python', 'r_studio', 'spark', 'aws', 'excel', 'avg_salary', 'Size_-1', 
                   'Size_1 to 50 employees', 'Size_10000+ employees', 'Size_1001 to 5000 employees', 
                   'Size_201 to 500 employees', 'Size_5001 to 10000 employees', 'Size_501 to 1000 employees', 
                   'Size_51 to 200 employees', 'Size_Unknown', 'Type of ownership_-1', 'Type of ownership_College / University', 
                   'Type of ownership_Company - Private', 'Type of ownership_Company - Public', 'Type of ownership_Government', 
                   'Type of ownership_Hospital', 'Type of ownership_Nonprofit Organization', 'Type of ownership_Other Organization',
                   'Type of ownership_School / School District', 'Type of ownership_Subsidiary or Business Segment', 
                   'Type of ownership_Unknown', 'Industry_-1', 'Industry_Accounting', 'Industry_Advertising & Marketing', 
                   'Industry_Aerospace & Defense', 'Industry_Architectural & Engineering Services', 'Industry_Auctions & Galleries',
                   'Industry_Banks & Credit Unions', 'Industry_Beauty & Personal Accessories Stores', 'Industry_Biotech & Pharmaceuticals',
                   'Industry_Brokerage Services', 'Industry_Colleges & Universities', 'Industry_Computer Hardware & Software',
                   'Industry_Construction', 'Industry_Consulting', 'Industry_Consumer Product Rental', 
                   'Industry_Consumer Products Manufacturing', 'Industry_Department, Clothing, & Shoe Stores',
                   'Industry_Education Training Services', 'Industry_Energy', 'Industry_Enterprise Software & Network Solutions', 
                   'Industry_Farm Support Services', 'Industry_Federal Agencies', 'Industry_Financial Analytics & Research', 
                   'Industry_Financial Transaction Processing', 'Industry_Food & Beverage Manufacturing', 'Industry_Gambling',
                   'Industry_Gas Stations', 'Industry_Health Care Products Manufacturing', 'Industry_Health Care Services & Hospitals',
                   'Industry_Health, Beauty, & Fitness', 'Industry_IT Services', 'Industry_Industrial Manufacturing', 
                   'Industry_Insurance Agencies & Brokerages', 'Industry_Insurance Carriers', 'Industry_Internet',
                   'Industry_Investment Banking & Asset Management', 'Industry_K-12 Education', 'Industry_Lending', 
                   'Industry_Logistics & Supply Chain', 'Industry_Metals Brokers', 'Industry_Mining',
                   'Industry_Motion Picture Production & Distribution', 'Industry_Other Retail Stores', 'Industry_Real Estate',
                   'Industry_Religious Organizations', 'Industry_Research & Development', 'Industry_Security Services', 
                   'Industry_Social Assistance', 'Industry_Sporting Goods Stores', 'Industry_Staffing & Outsourcing',
                   'Industry_Stock Exchanges', 'Industry_TV Broadcast & Cable Networks', 'Industry_Telecommunications Manufacturing',
                   'Industry_Telecommunications Services', 'Industry_Transportation Equipment Manufacturing', 
                   'Industry_Transportation Management', 'Industry_Travel Agencies', 'Industry_Trucking', 'Industry_Video Games', 
                   'Industry_Wholesale', 'Sector_-1', 'Sector_Accounting & Legal', 'Sector_Aerospace & Defense', 
                   'Sector_Agriculture & Forestry', 'Sector_Arts, Entertainment & Recreation', 'Sector_Biotech & Pharmaceuticals', 
                   'Sector_Business Services', 'Sector_Construction, Repair & Maintenance', 'Sector_Consumer Services', 
                   'Sector_Education', 'Sector_Finance', 'Sector_Government', 'Sector_Health Care', 'Sector_Information Technology', 
                   'Sector_Insurance', 'Sector_Manufacturing', 'Sector_Media', 'Sector_Mining & Metals', 'Sector_Non-Profit', 
                   'Sector_Oil, Gas, Energy & Utilities', 'Sector_Real Estate', 'Sector_Retail', 'Sector_Telecommunications',
                   'Sector_Transportation & Logistics', 'Sector_Travel & Tourism', 'Revenue_$1 to $2 billion (USD)', 
                   'Revenue_$1 to $5 million (USD)', 'Revenue_$10 to $25 million (USD)', 'Revenue_$10+ billion (USD)', 
                   'Revenue_$100 to $500 million (USD)', 'Revenue_$2 to $5 billion (USD)', 'Revenue_$25 to $50 million (USD)',
                   'Revenue_$5 to $10 billion (USD)', 'Revenue_$5 to $10 million (USD)', 'Revenue_$50 to $100 million (USD)', 
                   'Revenue_$500 million to $1 billion (USD)', 'Revenue_-1', 'Revenue_Less than $1 million (USD)', 
                   'Revenue_Unknown / Non-Applicable', 'title_simp_data analyst', 'title_simp_data engineer', 
                   'title_simp_data scientist', 'title_simp_director', 'title_simp_manager', 'title_simp_mle', 'title_simp_na',
                   'seniority_jr', 'seniority_na', 'seniority_senior']

newdata = [[3.8,'501 to 1000 employees','Company - Private','Aerospace & Defense',
            'Aerospace & Defense','$50 to $100 million (USD)',0,0,0,48,1,0,0,0,1,'data scientist','na']]

data_df = pd.DataFrame(newdata, columns=columns_model)
data_dummies = data_df.reindex(labels = columns_dummies, axis = 1, fill_value = 0).drop(columns='avg_salary').values
data_in = data_dummies.tolist()