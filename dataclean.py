# importing needed packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# setting up names for columns
colnames = ['AccountStatus','Duration','CreditHistory','Purpose','CreditAmount',
            'Savings','EmploymentDuration','InstallmentRatePercent',
            'PersonalStatusAndSex','OtherDebtors','ResidenceDuration',
            'Property','Age','OtherInstallment','Housing','CreditsInThisBank','Job',
            'NumberOfPeopleLiableToMaintenance','Telephone','Foreign','Decision']

# importing data
data = pd.read_csv(r"data\german.data",delimiter=" ",
                   names=colnames)
# changing data to dataframe
df = pd.DataFrame(data)

# changing values to be more readable for every column
df['AccountStatus'].replace({'A11':'0 DM',
                             'A12':'<200 DM',
                             'A13':'>=200 DM',
                             'A14':'No account'},inplace=True)

df['CreditHistory'].replace({'A30':'No credits taken',
                             'A31':'All credits paid duly',
                             'A32':'Existing credits paid duly till now',
                             'A33':'Delay in paying in past',
                             'A34':'Has credits in other banks'},inplace=True)

df['Purpose'].replace({'A40':'new car',
                       'A41':'used car',
                       'A42':'furnitures',
                       'A43':'radio/TV',
                       'A44':'household appliances',
                       'A45':'repairs',
                       'A46':'education',
                       'A47':'vacation',
                       'A48':'retraining',
                       'A49':'business',
                       'A410':'others'},inplace=True)

df['Savings'].replace({'A61':'<100 DM',
                       'A62':'100-500 DM',
                       'A63':'500-1000 DM',
                       'A64':'>1000 DM',
                       'A65':'Unknown/No savings'},inplace=True)

df['EmploymentDuration'].replace({'A71':'unemployed',
                                  'A72':'employed under 1 year',
                                  'A73':'employed from 1 to 4 years',
                                  'A74':'employed from 4 to 7 years',
                                  'A75':'employed 7 years or more'}, inplace=True)

df['PersonalStatusAndSex'].replace({'A91':'male divorced/separated',
                                    'A92':'female divorced/separated/married',
                                    'A93':'male single',p
                                    'A94':'male married/widowed',
                                    'A95':'female single'}, inplace=True)

df['OtherDebtors'].replace({'A101':'none',
                            'A102':'co-applicant',
                            'A103':'guarantor'}, inplace=True)

df['Property'].replace({'A121':'real estate',
                        'A122':'building society savings agreement/life insurance',
                        'A123':'car or other, not in attribute 6',
                        'A124':'unknown/no property'}, inplace=True)

df['OtherInstallment'].replace({'A141':'bank',
                                'A142':'stores',
                                'A143':'none'}, inplace=True)

df['Housing'].replace({'A151':'rent',
                       'A152':'own',
                       'A153':'for free'}, inplace=True)

df['Job'].replace({'A171':'unemployed/unskilled - non-resident',
                   'A172':'unskilled - resident',
                   'A173':'skilled employee/official',
                   'A174':'management/self-employed/highly qualified employee/officer'}, inplace=True)

df['Telephone'].replace({'A191':'none',
                         'A192':'yes'}, inplace=True)

df['Foreign'].replace({'A201':'yes',
                       'A202':'no'}, inplace=True)

# saving changed data to cleaned_data.csv
df.to_csv(r'C:\Users\Artur\Desktop\projekt\cleaned_data.csv')