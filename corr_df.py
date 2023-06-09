import seaborn as sns
import matplotlib.pyplot as plt

def corr_df(i,df):

    corr_df  = df.corr()

    plt.figure(figsize=(10,4),dpi=200)
    sns.barplot(x=corr_df[i].sort_values().iloc[1:-1].index,y=corr_df[i].sort_values().iloc[1:-1].values)
    plt.title(f"Feature Correlation to {i}")
    plt.xticks(rotation=90)
    plt.show()

    if len(df[f'{i}'].value_counts()) < 10:

        cluster_counts = df[f'{i}'].value_counts()
        plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')
        plt.show()

    else:

        sns.displot(data=df, x=f'{i}',kde=True,color='green',bins=20)
        plt.show()
