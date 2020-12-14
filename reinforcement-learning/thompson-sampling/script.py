import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

# gives the ad data with clicks per user
dataset = pd.read_csv('../../data/Ads_CTR_Optimisation.csv')

# N = len(dataset)  # total users
N = 300  # 300 iterations is enough to identify the ad with confidence
d = len(dataset.columns)  # number of ads
ads_selected = []
number_of_rewards_1 = [0] * d  # ad list for reward 1
number_of_rewards_0 = [0] * d  # ad list for reward 0
total_rewards = 0

for n in range(0, N):
  ad = 0

  max_random = 0
  for i in range(0, d):
    random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
    if (random_beta > max_random):
      # update the max_random
      max_random = random_beta
      ad = i

  ads_selected.append(ad)
  reward = dataset.values[n, ad]
  if reward == 1:
    number_of_rewards_1[ad] += 1
  else:
    number_of_rewards_0[ad] += 1

plt.hist(ads_selected)
plt.title('Histogram of Ads Selected')
plt.xlabel('Ads')
plt.ylabel('Number of Clicks on the Ad')
plt.show()
