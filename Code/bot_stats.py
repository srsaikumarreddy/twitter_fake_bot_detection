import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

def get_post_time_delta(group_df, deltas, retweets):
    group_df = group_df.sort_values("postedtime")
    posttimes = group_df["postedtime"].tolist()
    posttimes.insert(0, posttimes[0])
    del posttimes[-1]
    group_df["prev_post_time"] = posttimes
    group_df["post_time_delta"] = group_df.apply(lambda x: (x["postedtime"] - x["prev_post_time"]), axis=1).dt.total_seconds()
    retweets.append(group_df.tid.count())
    group_df = group_df[group_df["post_time_delta"] > 0]
    deltas.append(group_df.post_time_delta.median())

# retweet and post time plots
# actual_bots_2016_file = open("2018_actual_botnames.txt", 'r')
# detected_bots_2016_file = open("2018_detected_botnames_100_35_1.txt", 'r')
# actual_bots_2016 = [line.strip() for line in actual_bots_2016_file.readlines()]
# detected_bots_2016 = [line.strip() for line in detected_bots_2016_file.readlines()]
# deltas = []
# retweets = []
# df = pd.read_csv("swm-dataset/SWM-dataset.csv")
# a_df = df[df["postedtime"].str.contains("2018")]
# a_df = a_df[a_df["screen_name_from"].isin(actual_bots_2016)]
# #a_df = a_df[a_df["screen_name_from"].isin(detected_bots_2016)]
# a_df["postedtime"] = pd.to_datetime(a_df["postedtime"])
# a_df.sort_values(["screen_name_from", "retweet_tid", "postedtime"], inplace=True)
# g = a_df.groupby(["screen_name_from", "retweet_tid"]).filter(lambda g: g.retweet_tid.count() > 1).reset_index(drop=True)
# g.groupby(["screen_name_from", "retweet_tid"]).apply(lambda g: get_post_time_delta(g, deltas, retweets))
# deltas = [d / (3600 * 24) for d in deltas]
# s = pd.Series(deltas)
# r = pd.Series(retweets)
# a_df = a_df[a_df["screen_name_from"].isin(detected_bots_2016)]
# a_df["postedtime"] = pd.to_datetime(a_df["postedtime"])
# a_df.sort_values(["screen_name_from", "retweet_tid", "postedtime"], inplace=True)
# g = a_df.groupby(["screen_name_from", "retweet_tid"]).filter(lambda g: g.retweet_tid.count() > 1).reset_index(drop=True)
# g.groupby(["screen_name_from", "retweet_tid"]).apply(lambda g: get_post_time_delta(g, deltas, retweets))
# deltas = [d / (3600 * 24) for d in deltas]
# s2 = pd.Series(deltas)
# r2 = pd.Series(retweets)
# kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
# plt.figure(figsize=(10,7), dpi= 80)
# sns.distplot(r2, color="dodgerblue", label="Detected", hist=False, **dict(hist_kws={'alpha':.5}, kde_kws={'linewidth':2}))
# sns.distplot(r, color="orange", label="Listed", hist=False, **dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2}))
# plt.xlabel('Retweets', fontsize=18)
# plt.savefig('2018_retweets.png')

# suspended accounts

suspended_accounts_df = pd.read_csv("suspended.csv")
df = suspended_accounts_df
detected_bots_ds2_file = open("ds2_detected_botnames.txt", 'r')
detected_bots_ds2 = [line.strip() for line in detected_bots_ds2_file.readlines()]
df = df[df["name"].isin(detected_bots_ds2)]
df = df["suspended"].value_counts().reset_index()
df.columns = ["suspended", "frequency"]
#ax = df.plot.pie(y="frequency", labels=df.suspended, explode=(0.1, 0, 0), shadow=True, autopct='%1.0f%%', startangle=-154, figsize=(15,15))
ax = df.plot.pie(y="frequency", labels=["Found", "Suspended", "Not Found"], shadow=True, autopct='%1.0f%%', startangle=-154, figsize=(15,15))
ax.set_ylabel('')
plt.savefig('ds2_detected_suspended_pie2.png')
#print(df[(df["is_listed_bot"] == False) & (df["suspended"] == 1)].head())
#print(df[(df["is_listed_bot"] == True)].shape[0])





#df = pd.read_csv("swm-dataset/ds2.csv")
#names = df["screen_name_from"].unique().tolist()
#with open('ds2_names.txt', 'w') as f:
#    for botname in names:
#        f.write("%s\n" % botname)

# df = df["suspended"].value_counts().reset_index()
# df.columns = ["suspended", "frequency"]
# ax = df.plot.pie(y="frequency", labels=["Found", "Not Found", "Suspended"], colors=["deepskyblue", "lavender", "gold"], explode=(0.1, 0, 0), shadow=True, autopct='%1.0f%%', startangle=-154, figsize=(15,15))
# ax.set_ylabel('')
# plt.savefig('2018_actual_bots_suspended_pie.png')