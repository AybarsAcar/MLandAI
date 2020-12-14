import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori

# dataset has bunch of null values it is fine they indicate 0
dataset = pd.read_csv('../../data/Market_Basket_Optimisation.csv')

transactions = []
for i in range(0, len(dataset)):  # change that to dataset length
  transactions.append(
    [str(dataset.values[i, j]) for j in range(0, len(dataset.columns)) if not pd.isnull(dataset.values[i, j])])

# train the model on the dataset
# min_length == max_length == 2 b/c buy one get one free deal so exactly 2 elements
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2,
                max_length=2)

results = list(rules)
for rule in results:
  print(rule)


def inspect(results):
  lhs = [tuple(result[2][0][0])[0] for result in results]
  rhs = [tuple(result[2][0][1])[0] for result in results]
  supports = [result[1] for result in results]
  confidences = [result[2][0][2] for result in results]
  lifts = [result[2][0][3] for result in results]
  return list(zip(lhs, rhs, supports, confidences, lifts))


result_df = pd.DataFrame(inspect(results=results),
                         columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# sort according to Lift column
result_df.nlargest(n=len(result_df), columns='Lift')

print(result_df)
