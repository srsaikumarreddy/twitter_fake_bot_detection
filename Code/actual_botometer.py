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
df = pd.read_csv(path + '/dataset/2016_actual_botnames.csv')

accounts = df['screen_name_from'].values.tolist()

bots_row = []
non_bots_row = []
not_found_row = []

bar = IncrementalBar('Countdown', max = len(accounts))
for screen_name, result in bom.check_accounts_in(accounts):
    bots = []
    non_bots = []
    not_found = []
    try:
      result = bom.check_account("@" + screen_name)
      if result['display_scores']['universal']['overall'] >= 3.5:
        bots.append(screen_name)
        bots_row.append(bots)
      else:
        non_bots.append(screen_name)
        non_bots_row.append(non_bots)
    except:
      not_found.append(screen_name)
      not_found_row.append(not_found)
    bar.next()
    time.sleep(3)
bar.finish()

header = ["screen_name_from"]
bot_data = "2016_bot_data.csv"
non_bot_data = "2016_non_bot_data.csv"
not_found_data = "2016_not_found_data.csv"
    
with open(bot_data, 'w') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(header) 
    csvwriter.writerows(bots_row)

with open(non_bot_data, 'w') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(header) 
    csvwriter.writerows(non_bots_row)

with open(not_found_data, 'w') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(header) 
    csvwriter.writerows(not_found_row)

print("Done")
