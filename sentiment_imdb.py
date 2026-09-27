from transformers import pipeline
import pandas as pd 

classifier = pipeline("sentiment-analysis") 

df = pd.read_csv("imdb_top_1000.csv")
df_sample = df.head(50)

labels = []
scores = []

for index, row in df_sample.iterrows():
    title = row["Series_Title"]
    overview = row["Overview"]
    result = classifier(overview)
    labels.append(result[0]["label"])
    scores.append(result[0]["score"])   


df_sample["sentiment_label"] = labels
df_sample["sentiment_score"] = scores

print(df_sample[["Series_Title", "IMDB_Rating", "sentiment_label", "sentiment_score"]])

print(df_sample["sentiment_label"].value_counts())
print(df_sample.groupby("sentiment_label")["IMDB_Rating"].mean())
