#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("./tone_responses.csv")
#%%
#cleaning 
df.replace('"', '', inplace=True)
df.iloc[63]


#%%

#right now our df has text data not numerical data
#we need to convert the text data into numerical data for knn to work

#this map converts text to numerical values
df["tone"] = df["tone"].map({
    "neutral": 0,
    "curious": 1,
    "happy": 2,
    "excited": 3,
    "bored": 4,
    "confused": 5
})


#the vectorizer converts text into numerical data using the tf-idf method
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["comment"])
y = df["tone"]




def minmax(x):
    u= (x-min(x))/(max(x)-min(x))
    return u



sns.scatterplot(df, x="comment", y="tone")
plt.xticks(rotation=90)  
plt.show()

#%%

pd.crosstab(y, y_hat)

#%%
model = KNeighborsClassifier()
model = model.fit(minmax(x=))
y_hat = model.predict(u_test)
# %%
