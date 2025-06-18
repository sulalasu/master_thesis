import pandas as pd


my_df = pd.DataFrame({
    "a" : [1, 2, 3],
    "b" : [4, 5, 6],
    "c" : [7, 8, 9]
})

def merge_columns(df, columns_to_merge, colname):
    df[colname] = [11, 22, 33]
    print(df)

merge_columns(my_df, "test", "newcols")



plt.plot("0", "13", data=seoul)
plt.show
