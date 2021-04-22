import botometer
import os.path
import pandas as pd
import csv
import time
from progress.bar import IncrementalBar  

rapidapi_key = ""
twitter_app_auth = {
    'consumer_key': '',
    'consumer_secret': '',
    'access_token': '',
    'access_token_secret': '',
  }
bom = botometer.Botometer(wait_on_ratelimit=True,
                          rapidapi_key=rapidapi_key,
                          **twitter_app_auth)

path = os.getcwd()
full_df = pd.read_csv(path + '/dataset/full/SWM-dataset.csv')
detected_df = pd.read_csv(path + '/dataset/detected/2016_detected_botnames_100_35_1.csv')
actual_bots_df = pd.read_csv(path + '/dataset/actual/2016_actual_botnames.csv')

detected_size = len(detected_df)
print("Detected bots: " + str(detected_size))

detected_df = detected_df.loc[~((detected_df.screen_name_from.isin(actual_bots_df['screen_name_from']))),:]
detected_minus_actual_size = len(detected_df)
print("Detected bots after actual bots removal: " + str(detected_minus_actual_size))
actual_bots_in_sample = detected_size - detected_minus_actual_size
print("Actual bots in sample: " + str(actual_bots_in_sample))

retweets_df = pd.DataFrame(full_df['screen_name_from'].value_counts())
retweets_df.reset_index(level=0, inplace=True)
retweets_df.columns = ['screen_name_from', 'retweet_count']

retweets_df = retweets_df.loc[((retweets_df.screen_name_from.isin(detected_df['screen_name_from']))),:]
retweets_df_chunk1 = retweets_df.head(1000)
retweets_df_chunk2 = retweets_df.iloc[1000:2000]
retweets_df_chunk3 = retweets_df.iloc[2000:]

accounts = retweets_df_chunk2['screen_name_from'].values.tolist()

bots_row = []
not_found = []
def_bots = 0
maybe_bots = 0
not_bots = 0

bar = IncrementalBar('Countdown', max = len(accounts))
for screen_name, result in bom.check_accounts_in(accounts):
    bots = [] 
    try:
      result = bom.check_account("@" + screen_name)
      score = result['display_scores']['universal']['overall']
      if score >= 4:
        def_bots += 1
      elif score >= 3.5 and score < 4:
        maybe_bots += 1
      else:
        not_bots += 1
      bots.append(screen_name)
      bots.append(score)
      bots_row.append(bots)
    except:
      not_found.append(screen_name)
    bar.next()
    time.sleep(2)
bar.finish()

header = ["screen_name_from", "score"]
bot_data = "2016_bot_data2.csv"
    
with open(bot_data, 'w') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(header) 
    csvwriter.writerows(bots_row)

print("Def bots: " + str(def_bots))
print("Maybe bots: " + str(maybe_bots))
print("Not bots: " + str(not_bots))
print("Not found: " + str(len(not_found)))
