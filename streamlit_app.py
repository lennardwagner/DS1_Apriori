import numpy as np
import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules

@st.cache
def load_data():
    """
    Load and transform data.
    :return:
    """
    df = pd.read_csv("autism_data.csv")
    df.replace("?", np.NaN, inplace=True)

    # remove columns with missing values
    df.dropna(axis=1, how="any", inplace=True)
    df.drop("age_desc", axis=1, inplace=True)
    new_df = df.copy()

    # Create dummies and drop original columns
    df_dummies = pd.get_dummies(df,
                                columns=['age', 'gender', 'jundice', 'autism',
                                         'country_of_res', 'used_app_before',
                                         'result_numeric', 'class/asd'])
    new_df.drop(
        ["age", "gender", "jundice", "country_of_res", "used_app_before",
         "result_numeric"], axis=1, inplace=True)

    new_df_dummies = pd.get_dummies(new_df, columns=['autism', 'class/asd'])
    return new_df_dummies


@st.cache
def run_apriori(data, min_sup):
    frequent_items = apriori(data, min_sup, use_colnames=True)
    frequent_items = pd.DataFrame(frequent_items)
    #print(frequent_items["itemsets"])
    frequent_items["itemsets"] = frequent_items["itemsets"].apply(list)
    return frequent_items


def run_association_rules(data, min_threshold, min_sup):
    frequent_items = apriori(data, min_sup, use_colnames=True)
    association = association_rules(frequent_items, metric="confidence", min_threshold=min_threshold)
    association = pd.DataFrame(association)
    association["antecedents"] = association["antecedents"].apply(list)
    association["consequents"] = association["consequents"].apply(list)
    return association


if __name__ == "__main__":
    st.title("Apriori Algorithm")
    raw_data = st.sidebar.checkbox("Show raw data", value=True)
    show_equations = st.sidebar.checkbox("Show equations", value=False)
    min_support = st.sidebar.slider("Minimum support value",
                            min_value=0.0,
                            max_value=1.0)
    min_threshold = st.sidebar.slider("Minimum threshold value",
                            min_value=0.0,
                            max_value=1.0)

    if raw_data:
        df = pd.read_csv("autism_data.csv")
        st.write(df)

    data = load_data()
    #data.astype(str)
    st.write("Transformed data")
    st.dataframe(data)

    st.write("Frequent items")
    frequent_items = run_apriori(data, min_support)
    st.dataframe(frequent_items)
    #association = run_association_rules(data, min_threshold, min_support)

    st.write("Association rules")
    if show_equations:
        #st.write("Confidence:")
        st.latex(r"confidence(X\Rightarrow Y)=\frac{support(X\bigcap_{}^{}Y))}{support(X))}")
        #st.write("Lift:")
        st.latex(r"lift(X\Rightarrow Y)=\frac{support(X\bigcap_{}^{}Y))}{support(X)\times support(Y)}")
        #st.write("Leverage:")
        st.latex(r"leverage(X\Rightarrow Y)=support(X\cap Y) - support(X) \times support(Y)")
        #st.write("Conviction:")
        st.latex(r"conviction(X\Rightarrow Y)=\frac{1-support(Y)}{1-confidence(X\Rightarrow Y)}")

    association = run_association_rules(data, min_threshold, min_support)
    st.dataframe(association)