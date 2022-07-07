import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
import pydotplus
from scipy.stats import entropy
import math
dataset_url = "https://raw.githubusercontent.com/marcus-zuba/Musical-genres-classification/main/songs.csv"
dataset = pd.read_csv(dataset_url)
dataset = dataset[dataset.genre != "set()"]
dataset
#2
number_of_rows = len(dataset.index)
genres = []

for genre in dataset['genre'].unique():
  splited_genres = genre.split(',')
  if splited_genres[0].strip() not in dataset:
    dataset[splited_genres[0].strip()] = np.zeros(number_of_rows)
    genres.append(splited_genres[0].strip())

for index, row in dataset.iterrows():
  genres_for_row = row['genre'].split(',')
  dataset.loc[index, genres_for_row[0].strip()] = 1


artists = dataset['artist'].unique()

for i, artist in enumerate(artists):
  dataset.loc[dataset['artist'] == artist, 'artist'] = i

dataset = dataset.drop('song',axis=1)
dataset = dataset.drop('genre',axis=1)
# dataset[['artist','genre','rock','pop']]
dataset

#3
genres_percentages = {}
for genre in genres:
  class_proportion = round((len(dataset[dataset[genre] == 1].index)/number_of_rows) * 100, 2)
  genres_percentages[genre] = class_proportion

genres_percentages

#4
dataset = dataset[(dataset['pop'] == 1) | (dataset['hip hop'] == 1) | (dataset['rock'] == 1) | (dataset['Dance/Electronic'] == 1)]
for genre in genres:
  if genre != 'pop' and genre != 'hip hop' and genre != 'rock' and genre != 'Dance/Electronic':
    dataset = dataset.drop(genre, axis = 1)

genres_percentages = {}
number_of_rows = len(dataset.index)
for genre in genres:
  if genre in dataset:
    class_proportion = round((len(dataset[dataset[genre] == 1].index)/number_of_rows) * 100, 2)
    genres_percentages[genre] = class_proportion

genres_percentages

#5
from math import floor
# set training data frame

df_size = int(number_of_rows / 2)
rows_for_classes = {}
for key, value in genres_percentages.items():
  rows_for_class = int(value * floor(df_size) / 100)
  rows_for_classes[key] = rows_for_class

df_training = pd.DataFrame()

for style,rows in rows_for_classes.items():
  filtered_data = dataset[dataset[style] == 1]
  number_of_rows_to_get = rows
  if rows > len(filtered_data.index):
    number_of_rows_to_get = len(filtered_data.index)
  selected = filtered_data.sample(number_of_rows_to_get)
  df_training = pd.concat([df_training, selected])

# set test data frame
df_test = pd.DataFrame()

remaining_df = pd.concat([df_training,dataset]).drop_duplicates(keep=False)
remaining_df_size = len(remaining_df.index)

if df_size > remaining_df_size:
  for key, value in genres_percentages.items():
    rows_for_class = int(value * floor(remaining_df_size) / 100)
    rows_for_classes[key] = rows_for_class


for style,rows in rows_for_classes.items():
  filtered_data = remaining_df[remaining_df[style] == 1]
  number_of_rows_to_get = rows
  if rows > len(filtered_data.index):
    number_of_rows_to_get = len(filtered_data.index)
  selected = filtered_data.sample(number_of_rows_to_get)
  df_test = pd.concat([df_test, selected])

# print(df_size)
# print(len(df_training.index))
# print(len(df_test.index))

df_training.to_csv('training.csv',index=False)
df_test.to_csv('test.csv',index=False)