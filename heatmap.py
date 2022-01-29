from sklearn import datasets
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

iris=datasets.load_iris()
data=iris.data

df=pd.DataFrame(data, columns=iris.feature_names)
map=sns.heatmap(df.corr(method="pearson"), annot=True)
plt.xticks(rotation=90)
plt.show()