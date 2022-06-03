import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("autism_data.csv")
df.replace("?", np.NaN, inplace=True)
#print(df)
df.dropna(axis=1, how="any", inplace=True)
df.drop("age_desc", axis=1, inplace=True)
new_df = df.copy()
#print(df)
df_dummies = pd.get_dummies(df, columns=['age', 'gender', 'jundice', 'autism', 'country_of_res', 'used_app_before', 'result_numeric', 'class/asd'])
#print(df_dummies.head())
new_df.drop(["age", "gender", "jundice", "country_of_res", "used_app_before", "result_numeric"], axis=1, inplace=True)
#print(new_df)
new_df_dummies = pd.get_dummies(new_df, columns=['autism', 'class/asd'])
#print(new_df_dummies)

'''b = apriori_new(df_dummies)
for item in b:
    print(f"{item}: {b[item]}")'''


# apriori
frequent_items = apriori(df_dummies, min_support=0.70, use_colnames=True)
association_rules = association_rules(frequent_items, min_threshold=0.8)

#association_rules.to_csv("results.txt", sep=" ", mode="a")
print(association_rules)


