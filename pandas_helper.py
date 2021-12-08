import pandas as pd


df = pd.read_csv("data/hyperparam_tuning_rb_acrobot.csv", delimiter=",", index_col="Unnamed: 0")
print(df)
df.to_html("data/hyperparam_tuning_acrobot.html")