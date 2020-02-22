import statsmodels.api as sm
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from fancyimpute import IterativeImputer

df = pd.read_csv('pima.csv')
msno.bar(df)
msno.heatmap(df)

df_cleaned = df.dropna(subset=['Diastolic_BP', 'BMI', 'Glucose'])
df_noNa = df.dropna()
y_noNa = df_noNa['Class']
X_noNa = df_noNa.iloc[:, :-1]
lm_noNa = sm.OLS(y_noNa, X_noNa).fit()
R2 = pd.Series([lm_noNa.rsquared_adj, 0, 0], index=['noNa', 'MICE', 'medians'])

df_MICE = df_cleaned.copy(deep=True)
MICE_imputer = IterativeImputer()
df_MICE.iloc[:, :] = MICE_imputer.fit_transform(df_MICE)

y_MICE = df_MICE['Class']
X_MICE = df_MICE.iloc[:, :-1]
lm_MICE = sm.OLS(y_MICE, X_MICE).fit()
lm.summary()
R2['MICE'] = lm_MICE.rsquared_adj

df_medians = df_cleaned.copy(deep=True)
df_medians.loc[df_medians['Serum_Insulin'].isna(), 'Serum_Insulin'] = df_medians['Serum_Insulin'].median()
df_medians.loc[df_medians['Skin_Fold'].isna(), 'Skin_Fold'] = df_medians['Skin_Fold'].median()
y_medians = df_medians['Class']
X_medians = df_medians.iloc[:, :-1]
lm_medians = sm.OLS(y_medians, X_medians).fit()
R2['medians'] = lm_medians.rsquared_adj


df_MICE['Serum_Insulin'].plot(kind='kde', label='MICE')
df['Serum_Insulin'].plot(kind='kde', label='original')
df_medians['Serum_Insulin'].plot(kind='kde', label='medians')
plt.legend()
plt.show()

print(R2)
