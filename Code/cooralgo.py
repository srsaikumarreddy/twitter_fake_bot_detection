import csv
import time
import tweepy
import pandas as pd
import numpy as np
from tqdm import tqdm


# filter out tweets based on retweets
def filter_out_tweets_with_retweet_threshold(dataset_df, retweet_threshold):
    dataset_df["total_retweets"] = dataset_df.groupby("retweet_tid")["retweet_tid"].transform("count")
    dataset_df.drop(dataset_df[dataset_df["total_retweets"] < retweet_threshold].index, inplace=True)
    del dataset_df["total_retweets"]


# calculates coordination interval
def calculate_coordination_interval(tweet_df, p, q):
    tweet_df["postedtime"] = pd.to_datetime(tweet_df["postedtime"])
    tweet_df["unique_retweets"] = tweet_df.groupby("retweet_tid")["tid"].transform("nunique")
    tweet_df["first_posttime"] = tweet_df.groupby("retweet_tid")["postedtime"].transform("min")
    tweet_df["last_posttime"] = tweet_df.groupby("retweet_tid")["postedtime"].transform("max")
    tweet_df["rank"] = tweet_df.groupby("retweet_tid")["postedtime"].rank(method="first")
    tweet_df["percentage_of_retweets"] = tweet_df["rank"] / tweet_df["unique_retweets"]
    tweet_df["time_delta_from_first_posttime"] = (tweet_df["postedtime"] - tweet_df["first_posttime"]).dt.total_seconds()
    tweet_df = tweet_df.sort_values("retweet_tid")
    rank_2_tweet_df = tweet_df[tweet_df["rank"] == 2].copy()
    rank_2_tweet_df = rank_2_tweet_df[rank_2_tweet_df["time_delta_from_first_posttime"] <= rank_2_tweet_df["time_delta_from_first_posttime"].quantile(q)]
    tweet_subset_df = tweet_df[tweet_df.retweet_tid.isin(rank_2_tweet_df.retweet_tid)].copy()
    tweet_subset_df = tweet_subset_df[tweet_subset_df["percentage_of_retweets"] > p]
    tweet_subset_df["time_delta_from_first_posttime"] = tweet_subset_df.groupby("retweet_tid")["time_delta_from_first_posttime"].transform("min")
    tweet_subset_df = tweet_subset_df[["retweet_tid", "time_delta_from_first_posttime"]].drop_duplicates()
    coordination_interval = tweet_subset_df["time_delta_from_first_posttime"].quantile(p)
    if coordination_interval == 0:
        return 1
    return coordination_interval


# find bots based on coordination interval calculated
def apply_first_filter(group_df, coordination_interval, pbar):
    group_df = group_df.sort_values("postedtime")
    posttimes = group_df["postedtime"].tolist()
    posttimes.insert(0, posttimes[0])
    del posttimes[-1]
    group_df["prev_post_time"] = posttimes
    group_df["post_time_delta"] = group_df.apply(lambda x: (x["postedtime"] - x["prev_post_time"]), axis=1).dt.total_seconds()
    prev_screen_names = group_df["screen_name_from"].tolist()
    prev_screen_names.insert(0, prev_screen_names[0])
    del prev_screen_names[-1]
    group_df["prev_screen_name_from"] = prev_screen_names
    group_df = group_df[group_df["post_time_delta"] <= coordination_interval]
    if group_df.shape[0] > 1:
        group_df = group_df[group_df["screen_name_from"] != group_df["prev_screen_name_from"]]
        candidates = group_df["screen_name_from"].unique().tolist()
        candidates.extend(group_df["prev_screen_name_from"].unique().tolist())
        pbar.update(1)
        return list(set(candidates))
    else:
        pbar.update(1)


# apply two filters on dataset to detect bots
def find_bots(tweet_df, coordination_interval, retweet_count_threshold):    
    print("applying first filter based on coordination interval...\n")
    groups = [group  for name, group in tweet_df.groupby("retweet_tid")]
    with tqdm(total=len(groups)) as pbar:
        name_list = [apply_first_filter(g, coordination_interval, pbar) for g in groups]
    name_list = [names for names in name_list if names is not None]
    name_list = list(set([name for sublist in name_list for name in sublist]))
    print("\napplying second filter based on minimum retweet threshold...")
    df = tweet_df[tweet_df["screen_name_from"].isin(name_list)]
    second_filter = df.groupby("screen_name_from")["retweet_tid"].count()
    print("done")
    return second_filter[second_filter >= retweet_count_threshold].index.tolist()


# basic evaluation measures
def run_evaluation_measures(tweet_df, detected_botnames, actual_botnames):
    correctly_classified_as_bots_botnames = set(detected_botnames).intersection(actual_botnames)
    incorrectly_classified_as_bots_names = set(detected_botnames).difference(correctly_classified_as_bots_botnames)
    incorrectly_classified_as_not_bots_botnames = set(actual_botnames).difference(correctly_classified_as_bots_botnames)
    tp = len(correctly_classified_as_bots_botnames)
    fp = len(incorrectly_classified_as_bots_names)
    fn = len(incorrectly_classified_as_not_bots_botnames)
    tn = tweet_df["screen_name_from"].nunique() - len(actual_botnames) - fp
    print(f"actual bots: {len(actual_botnames)}")
    print(f"detected bots: {len(detected_botnames)}\n")
    print(f"precision: {round(tp/(tp+fp), 3)}")
    print(f"recall: {round(tp/(tp+fn), 3)}")
    print(f"accuracy: {round((tp+tn)/(tp+tn+fp+fn), 3)}")


def create_graph_csv_files(tweet_df):
    edge_df = tweet_df.copy()
    edge_df["retweet_count"] = edge_df.groupby(["screen_name_from", "screen_name_to"], as_index=False)["screen_name_to"].transform("count")
    edge_df = edge_df[["screen_name_from", "screen_name_to", "retweet_count"]]
    edge_df.columns = ["Source", "Target", "Weight"]
    edge_df = edge_df.drop_duplicates(["Source", "Target"])
    edge_df.to_csv("graph/edges_2016.csv", index=False)
    nodes_df = tweet_df.groupby("screen_name_from").first().reset_index()[["screen_name_from", "is_listed_bot"]]
    nodes_df.columns = ["screen_name_from", "class"]
    nodes_df["class"] = nodes_df["class"].apply(lambda x: "Listed" if x else "Detected")
    screen_name_to = {"screen_name_from": tweet_df["screen_name_to"].unique().tolist()}
    nodes_df = nodes_df.append(pd.DataFrame(screen_name_to))
    nodes_df.drop_duplicates(subset=["screen_name_from"], inplace=True)
    nodes_df.columns = ["Id", "Class"]
    nodes_df.fillna(value="Normal", inplace=True)
    nodes_df.to_csv("graph/nodes_2016.csv", index=False)


def create_graph_csv_files_ds2(tweet_df):
    edge_df = tweet_df.copy()
    edge_df["retweet_count"] = edge_df.groupby(["screen_name_from", "screen_name_to"], as_index=False)["screen_name_to"].transform("count")
    edge_df = edge_df[["screen_name_from", "screen_name_to", "retweet_count"]]
    edge_df.columns = ["Source", "Target", "Weight"]
    edge_df = edge_df.drop_duplicates(["Source", "Target"])
    edge_df.to_csv("graph/edges_ds2.csv", index=False)
    screen_name_from = tweet_df["screen_name_from"].unique().tolist()
    screen_name_to = tweet_df["screen_name_to"].unique().tolist()
    screen_name_to = list(set(screen_name_to).difference((set(screen_name_from))))
    screen_name_to = {"screen_name_from": screen_name_to}
    nodes_df = pd.DataFrame(screen_name_from, columns=["screen_name_from"])
    nodes_df["class"] = np.nan
    nodes_df.fillna(value="Detected", inplace=True)
    nodes_df = nodes_df.append(pd.DataFrame(screen_name_to))
    nodes_df.drop_duplicates(inplace=True)
    nodes_df.fillna(value="Normal", inplace=True)
    nodes_df.columns = ["Id", "Class"]
    nodes_df.to_csv("graph/nodes_ds2.csv", index=False)


def run_ds1():
    # tunable parameters
    p = 0.35
    q = 0.1
    second_filter_retweet_threshold = 100
    minimum_retweets_per_tweet_in_dataset = 2
    
    # write bots to file flag
    write_actual_botnames_to_file = False

    # load dataset from file
    tweet_dataset_df = pd.read_csv('swm-dataset/SWM-dataset.csv')
    botname_df = pd.read_csv('swm-dataset/botnames.csv')

    # preprocess data
    # split dataset by year
    tweet_subset_a_df = tweet_dataset_df[tweet_dataset_df["postedtime"].str.contains("2016")]
    #tweet_subset_b_df = tweet_dataset_df[tweet_dataset_df["postedtime"].str.contains("2018")]

    # copy datasets
    tweet_subset_a_copy_df = tweet_subset_a_df.copy()
    #tweet_subset_b_copy_df = tweet_subset_b_df.copy()

    # split botnames by year
    botname_set = set(botname_df["BotName"].unique())
    subset_a_from_screen_names = set(tweet_subset_a_df["screen_name_from"].unique())
    #subset_b_from_screen_names = set(tweet_subset_b_df["screen_name_from"].unique())
    subset_a_botnames = list(botname_set.intersection(subset_a_from_screen_names))
    #subset_b_botnames = list(botname_set.intersection(subset_b_from_screen_names))

    # print sizes
    print(f"\nretweets in 2016 subset: {tweet_subset_a_df.shape[0]}")
    #print(f"retweets in 2018 subset: {tweet_subset_b_df.shape[0]}")
    print(f"bots in 2016 subset: {len(subset_a_botnames)}")
    #print(f"bots in 2018 subset: {len(subset_b_botnames)}")
    #print(f"common bots in both subsets: {len(set(subset_a_botnames).intersection(subset_b_botnames))}\n")

    # write actual botnames list to text file
    if write_actual_botnames_to_file:
        print("writing actual botnames to files\n")
        with open('2016_actual_botnames.txt', 'w') as f:
            for botname in subset_a_botnames:
                f.write("%s\n" % botname)
        with open('2018_actual_botnames.txt', 'w') as f:
            for botname in subset_b_botnames:
                f.write("%s\n" % botname)

    # get coordination interval for 2016 dataset
    coordination_interval_subset_a = calculate_coordination_interval(tweet_subset_a_df, p, q)
    print(f"coordination interval for 2016 subset: {coordination_interval_subset_a}")

    # remove redundant columns from 2016 dataframe
    del tweet_subset_a_df["tid"]
    del tweet_subset_a_df["screen_name_to"]
    del tweet_subset_a_df["unique_retweets"]
    del tweet_subset_a_df["rank"]
    del tweet_subset_a_df["percentage_of_retweets"]
    del tweet_subset_a_df["time_delta_from_first_posttime"]
    
    # get coordination interval for 2018 dataset
    #coordination_interval_subset_b = calculate_coordination_interval(tweet_subset_b_df, p, q)
    #print(f"coordination interval for 2018 subset: {coordination_interval_subset_b}\n")
    
    # remove redundant columns from 2018 dataframe
    #del tweet_subset_b_df["tid"]
    #del tweet_subset_b_df["screen_name_to"]
    #del tweet_subset_b_df["unique_retweets"]
    #del tweet_subset_b_df["rank"]
    #del tweet_subset_b_df["percentage_of_retweets"]
    #del tweet_subset_b_df["time_delta_from_first_posttime"]

    # remove single retweets from both datasets
    print(f"filtering retweets based on minimum retweet threshold\n")
    filter_out_tweets_with_retweet_threshold(tweet_subset_a_df, minimum_retweets_per_tweet_in_dataset)
    #filter_out_tweets_with_retweet_threshold(tweet_subset_b_df, minimum_retweets_per_tweet_in_dataset)

    # find bots in both datasets
    print(f"finding bots in 2016 dataset")
    subset_a_detected_botnames = find_bots(tweet_subset_a_df, coordination_interval_subset_a, second_filter_retweet_threshold)
    #print(f"\nfinding bots in 2018 dataset")
    #subset_b_detected_botnames = find_bots(tweet_subset_b_df, coordination_interval_subset_b, second_filter_retweet_threshold)

    # write identified botnames from 2016 to text files
    print(f"\nwriting 2016 detected botnames to file")
    with open('2016_detected_botnames.txt', 'w') as f:
        for botname in subset_a_detected_botnames:
            f.write("%s\n" % botname)
    
    # write identified botnames from 2018 to text files
    #print(f"writing 2018 detected botnames to file\n")
    #with open('2018_detected_botnames.txt', 'w') as f:
    #    for botname in subset_b_detected_botnames:
    #        f.write("%s\n" % botname)

    # run evaluation measures on both subsets
    print(f"running evaluation measures on 2016 dataset")   
    run_evaluation_measures(tweet_subset_a_df, subset_a_detected_botnames, subset_a_botnames)
    #print(f"\nrunning evaluation measures on 2018 dataset")
    #run_evaluation_measures(tweet_subset_b_df, subset_b_detected_botnames, subset_b_botnames)

    # filter original dataset by detected botnames only
    tweet_subset_a_copy_df = tweet_subset_a_copy_df[tweet_subset_a_copy_df["screen_name_from"].isin(subset_a_detected_botnames)]
    #tweet_subset_b_copy_df = tweet_subset_b_copy_df[tweet_subset_b_copy_df["screen_name_from"].isin(subset_b_detected_botnames)]
    tweet_subset_a_copy_df["is_listed_bot"] = tweet_subset_a_copy_df["screen_name_from"].isin(subset_a_botnames)
    #tweet_subset_b_copy_df["is_listed_bot"] = tweet_subset_b_copy_df["screen_name_from"].isin(subset_b_botnames)

    # create graph nodes and edges files
    print(f"Creating graph files for DS1-2016")
    create_graph_csv_files(tweet_subset_a_copy_df)
    
    #print(f"Creating graph files for DS1-2018")
    #create_graph_csv_files(tweet_subset_b_copy_df)


def run_ds2():
    # tunable parameters
    p = 0.4
    q = 0.1
    second_filter_retweet_threshold = 5
    minimum_retweets_per_tweet_in_dataset = 2

    # load dataset from file
    tweet_dataset_df = pd.read_csv('swm-dataset/ds2.csv')
    dataset_copy_df = tweet_dataset_df.copy()
    # print dataset size
    print(f"\nretweets in DS2: {tweet_dataset_df.shape[0]}")

    # get coordination interval for DS2
    coordination_interval = calculate_coordination_interval(tweet_dataset_df, p, q)
    print(f"coordination interval for DS2: {coordination_interval}")

    # remove redundant columns from 2016 dataframe
    del tweet_dataset_df["tid"]
    del tweet_dataset_df["screen_name_to"]
    del tweet_dataset_df["unique_retweets"]
    del tweet_dataset_df["rank"]
    del tweet_dataset_df["percentage_of_retweets"]
    del tweet_dataset_df["time_delta_from_first_posttime"]

    # remove single retweets from both datasets
    print(f"filtering retweets based on minimum retweet threshold\n")
    filter_out_tweets_with_retweet_threshold(tweet_dataset_df, minimum_retweets_per_tweet_in_dataset)

    # find bots in both datasets
    print(f"finding bots in DS2")
    detected_botnames = find_bots(tweet_dataset_df, coordination_interval, second_filter_retweet_threshold)
    
    print(f"Bots detected in DS2: {len(detected_botnames)}")

    # write identified botnames from DS2 to text files
    print(f"\nwriting DS2 detected botnames to file")
    with open('ds2_detected_botnames.txt', 'w') as f:
        for botname in detected_botnames:
            f.write("%s\n" % botname)
    
    # filter original dataset by detected botnames only
    dataset_copy_df = dataset_copy_df[dataset_copy_df["screen_name_from"].isin(detected_botnames)]
    
    # create graph nodes and edges files
    print(f"Creating graph files for DS2")
    create_graph_csv_files_ds2(dataset_copy_df)


def run_ds2_split():
    # load dataset
    tweet_dataset_df = pd.read_csv('swm-dataset/ds2.csv')
    tweet_subset_a_df = tweet_dataset_df[tweet_dataset_df["screen_name_to"] == "zerohedge"]
    tweet_subset_b_df = tweet_dataset_df[tweet_dataset_df["screen_name_to"] == "luiscarrillo66"]
    tweet_subset_c_df = tweet_dataset_df[tweet_dataset_df["screen_name_to"] == "zeusFanHouse"]
    tweet_subset_d_df = tweet_dataset_df[tweet_dataset_df["screen_name_to"] == "Mareq16"]
    tweet_subset_e_df = tweet_dataset_df[(tweet_dataset_df["screen_name_to"] != "Mareq16") & (tweet_dataset_df["screen_name_to"] != "zeusFanHouse") & (tweet_dataset_df["screen_name_to"] != "luiscarrillo66") & (tweet_dataset_df["screen_name_to"] != "zerohedge")]
    print(f"zerohedge first post: {tweet_subset_a_df['postedtime'].min()}, last post: {tweet_subset_a_df['postedtime'].max()}")
    print(f"luiscarrillo66 first post: {tweet_subset_b_df['postedtime'].min()}, last post: {tweet_subset_b_df['postedtime'].max()}")
    print(f"zeusFanHouse first post: {tweet_subset_c_df['postedtime'].min()}, last post: {tweet_subset_c_df['postedtime'].max()}")
    print(f"Mareq16 first post: {tweet_subset_d_df['postedtime'].min()}, last post: {tweet_subset_d_df['postedtime'].max()}")
    print(f"Other first post: {tweet_subset_e_df['postedtime'].min()}, last post: {tweet_subset_e_df['postedtime'].max()}")
    a_posters = set(tweet_subset_a_df['screen_name_from'].unique().tolist())
    b_posters = set(tweet_subset_b_df['screen_name_from'].unique().tolist())
    c_posters = set(tweet_subset_c_df['screen_name_from'].unique().tolist())
    d_posters = set(tweet_subset_d_df['screen_name_from'].unique().tolist())
    e_posters = set(tweet_subset_e_df['screen_name_from'].unique().tolist())
    print(f"zerohedge posters: {len(a_posters)}")
    print(f"luiscarrillo66 posters: {len(b_posters)}")
    print(f"zeusFanHouse posters: {len(c_posters)}")
    print(f"Mareq16 posters: {len(d_posters)}")
    print(f"Other posters: {len(e_posters)}")

    print(f"zerohedge tweeters overlaps: {len(a_posters.intersection(b_posters))}, {len(a_posters.intersection(c_posters))}, {len(a_posters.intersection(d_posters))}, {len(a_posters.intersection(e_posters))}")
    print(f"luiscarrillo66 tweeters overlaps: {len(b_posters.intersection(a_posters))}, {len(b_posters.intersection(c_posters))}, {len(b_posters.intersection(d_posters))}, {len(b_posters.intersection(e_posters))}")
    print(f"zeusFanHouse tweeters overlaps: {len(c_posters.intersection(a_posters))}, {len(c_posters.intersection(b_posters))}, {len(c_posters.intersection(d_posters))}, {len(c_posters.intersection(e_posters))}")
    print(f"Mareq16 tweeters overlaps: {len(d_posters.intersection(a_posters))}, {len(d_posters.intersection(b_posters))}, {len(d_posters.intersection(c_posters))}, {len(d_posters.intersection(e_posters))}")
    print(f"Other tweeters overlaps: {len(e_posters.intersection(a_posters))}, {len(e_posters.intersection(b_posters))}, {len(e_posters.intersection(c_posters))}, {len(e_posters.intersection(d_posters))}")

    p = 0.5
    q = 0.1
    second_filter_retweet_threshold = 5
    minimum_retweets_per_tweet_in_dataset = 5

    # get coordination interval for DS2
    coordination_interval_a = calculate_coordination_interval(tweet_subset_a_df, p, q)
    coordination_interval_b = calculate_coordination_interval(tweet_subset_b_df, p, q)
    coordination_interval_c = calculate_coordination_interval(tweet_subset_c_df, p, q)
    coordination_interval_d = calculate_coordination_interval(tweet_subset_d_df, p, q)
    coordination_interval_e = calculate_coordination_interval(tweet_subset_e_df, p, q)

    print(f"coordination interval for zerohedge: {coordination_interval_a}")
    print(f"coordination interval for luiscarrillo66: {coordination_interval_b}")
    print(f"coordination interval for zeusFanHouse: {coordination_interval_c}")
    print(f"coordination interval for Mareq16: {coordination_interval_d}")
    print(f"coordination interval for Other: {coordination_interval_e}")

    # remove single retweets from both datasets
    print(f"filtering retweets based on minimum retweet threshold\n")
    filter_out_tweets_with_retweet_threshold(tweet_subset_a_df, minimum_retweets_per_tweet_in_dataset)
    filter_out_tweets_with_retweet_threshold(tweet_subset_b_df, minimum_retweets_per_tweet_in_dataset)
    filter_out_tweets_with_retweet_threshold(tweet_subset_c_df, minimum_retweets_per_tweet_in_dataset)
    filter_out_tweets_with_retweet_threshold(tweet_subset_d_df, minimum_retweets_per_tweet_in_dataset)
    filter_out_tweets_with_retweet_threshold(tweet_subset_e_df, minimum_retweets_per_tweet_in_dataset)

    # find bots in both datasets
    print(f"finding bots for zerohedge")
    detected_botnames_a = find_bots(tweet_subset_a_df, coordination_interval_a, second_filter_retweet_threshold)
    print(f"Bots detected for zerohedge: {len(detected_botnames_a)}")

    print(f"finding bots for luiscarrillo66")
    detected_botnames_b = find_bots(tweet_subset_b_df, coordination_interval_b, second_filter_retweet_threshold)
    print(f"Bots detected for luiscarrillo66: {len(detected_botnames_b)}")

    print(f"finding bots for zeusFanHouse")
    detected_botnames_c = find_bots(tweet_subset_c_df, coordination_interval_c, second_filter_retweet_threshold)
    print(f"Bots detected for zeusFanHouse: {len(detected_botnames_c)}")

    print(f"finding bots for Mareq16")
    detected_botnames_d = find_bots(tweet_subset_d_df, coordination_interval_d, second_filter_retweet_threshold)
    print(f"Bots detected for Mareq16: {len(detected_botnames_d)}")

    print(f"finding bots for Other")
    detected_botnames_e = find_bots(tweet_subset_e_df, coordination_interval_e, second_filter_retweet_threshold)
    print(f"Bots detected for Other: {len(detected_botnames_e)}")

    detected_botnames = list(set(detected_botnames_a + detected_botnames_b + detected_botnames_c + detected_botnames_d + detected_botnames_e))

    # write identified botnames from DS2 to text files
    print(f"\nwriting DS2 detected botnames to file")
    with open('ds2_split_a_detected_botnames.txt', 'w') as f:
        for botname in detected_botnames_a:
            f.write("%s\n" % botname)
    with open('ds2_split_b_detected_botnames.txt', 'w') as f:
        for botname in detected_botnames_b:
            f.write("%s\n" % botname)
    with open('ds2_split_c_detected_botnames.txt', 'w') as f:
        for botname in detected_botnames_c:
            f.write("%s\n" % botname)
    with open('ds2_split_d_detected_botnames.txt', 'w') as f:
        for botname in detected_botnames_d:
            f.write("%s\n" % botname)
    with open('ds2_split_e_detected_botnames.txt', 'w') as f:
        for botname in detected_botnames_e:
            f.write("%s\n" % botname)
    with open('ds2_split_detected_botnames.txt', 'w') as f:
        for botname in detected_botnames:
            f.write("%s\n" % botname)
    
    # filter original dataset by detected botnames only
    dataset_copy_df = tweet_dataset_df[tweet_dataset_df["screen_name_from"].isin(detected_botnames)]
    
    # create graph nodes and edges files
    print(f"Creating graph files for DS2")
    create_graph_csv_files_ds2(dataset_copy_df)


def run_stats_for_sentiment_analysis():
    tweet_dataset_df = pd.read_csv('swm-dataset/ds2.csv')
    tweet_subset_c_df = tweet_dataset_df[tweet_dataset_df["screen_name_to"] == "zeusFanHouse"]
    zeus_bots = pd.read_csv('ds2_split_c_detected_botnames.txt', header=None)
    zeus_bots.columns = ["bots"]
    bot_subset = tweet_subset_c_df[tweet_subset_c_df["screen_name_from"].isin(zeus_bots["bots"])]
    bot_subset["retweet_count"] = bot_subset.groupby("retweet_tid")["tid"].transform("count")
    bot_subset["postedtime"]
    bot_subset.drop_duplicates(subset=["retweet_tid"], inplace=True)
    bot_subset.drop_duplicates(subset=["screen_name_from"], inplace=True)
    bot_subset.sort_values(by="retweet_count", ascending=False, inplace=True)
    bot_subset.to_csv("b.csv", index=False)
    print(bot_subset.head())
    bots = bot_subset["screen_name_from"].unique().tolist()
    with open('zeusFanHouse_bots.txt', 'w') as f:
        for botname in bots:
            f.write("%s\n" % botname)
    non_bot_subset = tweet_subset_c_df[~tweet_subset_c_df["screen_name_from"].isin(zeus_bots["bots"])]
    non_bot_subset["retweet_count"] = non_bot_subset.groupby("retweet_tid")["tid"].transform("count")
    non_bot_subset.drop_duplicates(subset=["retweet_tid"], inplace=True)
    non_bot_subset.drop_duplicates(subset=["screen_name_from"], inplace=True)
    non_bot_subset.sort_values(by="retweet_count", ascending=False, inplace=True)
    non_bot_subset.to_csv("nb.csv", index=False)
    nonbots = non_bot_subset["screen_name_from"].unique().tolist()
    print(non_bot_subset.head())
    with open('zeusFanHouse_non_bots.txt', 'w') as f:
        for botname in nonbots:
            f.write("%s\n" % botname)


def get_suspended_accounts(account_list):
    result = []
    api_key = ""
    api_secret_key = ""
    access_token = ""
    access_token_secret = ""
    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    with tqdm(total=len(account_list)) as pbar:
        with open('suspended.csv','a', buffering=1) as out:
            csv_out=csv.writer(out)
            csv_out.writerow(['name','suspended'])
            for account in account_list:
                try:
                    user = api.get_user(account, include_entities=False)
                    time.sleep(2)
                    csv_out.writerow((account, 0))
                except tweepy.TweepError as e:
                    if e.api_code == 63:
                        csv_out.writerow((account, 1))
                        print(f"{account} is suspended")
                    else:
                        print(e)
                        print(account)
                        csv_out.writerow((account, 2))
                finally:
                    pbar.update(1)


def run_get_suspended_accounts():
    accounts_file = open("ds2_names.txt", 'r')
    accounts = [line.strip() for line in accounts_file.readlines()]
    get_suspended_accounts(accounts)


if __name__ == '__main__':
    #run_ds1()
    run_ds2()
    #run_ds2_split()
    #run()
    #run_get_suspended_accounts()