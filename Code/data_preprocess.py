import pandas as pd
import os.path
import json
import csv
import time
from progress.bar import IncrementalBar

df = pd.DataFrame()
path = os.getcwd()
for chunk in pd.read_csv(path + "/dataset/full/tweets_dump.csv", iterator=True, engine='python', encoding='utf-8', error_bad_lines=False):
    df = pd.concat([df, chunk])
    
tweets = {}
for index, row in df.iterrows():
    json_object = json.loads(row['data'])
    tweets[row['tid']] = json_object
    
header = ["tid", "retweet_tid", "screen_name_from", "screen_name_to", "postedtime", "tweet_text", "retweet_text"]
rows = []
print("total tweets", len(tweets))
index = 0
bar = IncrementalBar('Progress', max = len(tweets))

for k in tweets:
    data = tweets[k]
    row_data = []
    if "quoted_status" in data:  
        row_data.append(str(data["id_str"]))
        row_data.append(str(data["quoted_status"]["id_str"]))
        row_data.append(str(data["user"]["screen_name"]))
        row_data.append(str(data["quoted_status"]["user"]["screen_name"]))
        row_data.append(str(data["created_at"]))
        row_data.append(str(data["text"]))
        if "extended_tweet" in data["quoted_status"]:
            row_data.append(str(data["quoted_status"]["extended_tweet"]["full_text"]))
        else:
            row_data.append(str(data["quoted_status"]["text"]))
    elif "retweeted_status" in data:
        row_data.append(str(data["id_str"]))
        row_data.append(str(data["retweeted_status"]["id_str"]))
        row_data.append(str(data["user"]["screen_name"]))
        row_data.append(str(data["retweeted_status"]["user"]["screen_name"]))
        row_data.append(str(data["created_at"]))
        row_data.append("")
        if "extended_tweet" in data["retweeted_status"]:
            row_data.append(str(data["retweeted_status"]["extended_tweet"]["full_text"]))
        else:
            row_data.append(str(data["retweeted_status"]["text"]))
    if row_data:
        rows.append(row_data)
    bar.next()
bar.finish()

filename = "processed_full_data.csv"
    
with open(filename, 'w') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(header) 
    csvwriter.writerows(rows)