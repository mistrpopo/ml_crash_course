import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
import os

# The ML course was done on TF 1.15 version (we are now at TF v2)
print(tf.__version__)
# 1.15.0

# TF fails to run if Cuda is available and the GPU device is not advanced enough :
# Ignoring visible gpu device (device: 0, name: NVS 5200M, pci bus id: 0000:01:00.0, compute capability: 2.1) with Cuda compute capability 2.1. The minimum required Cuda capability is 3.5.
# Disabling Cuda visible devices in order to run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

# was removed in TF v2
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("D:\\perso\\mistrpopo\\ml_crash_course\\california_housing_train.csv", sep=",")

# Randomize the data in order not to get any pathological ordering effects that might harm the performance of Stochastic Gradient Descent. 
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

def preprocess_features(california_housing_dataframe):
  """Prepares input features from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """

  # select all columns in the csv with the exception of the median_house_value (which is our target)
  selected_features = california_housing_dataframe[
    ["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]]
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  processed_features["rooms_per_person"] = (
    california_housing_dataframe["total_rooms"] /
    california_housing_dataframe["population"])
  
  return processed_features

def preprocess_targets(california_housing_dataframe):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["median_house_value"] = (
    california_housing_dataframe["median_house_value"] / 1000.0)
  return output_targets

# Choose the first 12000 (out of 17000) examples for training.
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

# Choose the last 5000 (out of 17000) examples for validation.
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

# Double-check that we've done the right thing.
print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())



correlation_dataframe = training_examples.copy()
correlation_dataframe["target"] = training_targets["median_house_value"]

# Correlation matrix : Pearson correlation coefficient measures how much features/targets are correlated to each other
# -1.0: perfect negative correlation
# 0.0: no correlation
# 1.0: perfect positive correlation
print(correlation_dataframe.corr())
#                     latitude  longitude  housing_median_age  total_rooms  total_bedrooms  population  households  median_income  rooms_per_person  target
# latitude                 1.0       -0.9                 0.0         -0.0            -0.1        -0.1        -0.1           -0.1               0.1    -0.1
# longitude               -0.9        1.0                -0.1          0.1             0.1         0.1         0.1           -0.0              -0.1    -0.0
# housing_median_age       0.0       -0.1                 1.0         -0.4            -0.3        -0.3        -0.3           -0.1              -0.1     0.1
# total_rooms             -0.0        0.1                -0.4          1.0             0.9         0.9         0.9            0.2               0.1     0.1
# total_bedrooms          -0.1        0.1                -0.3          0.9             1.0         0.9         1.0           -0.0               0.1     0.0
# population              -0.1        0.1                -0.3          0.9             0.9         1.0         0.9            0.0              -0.1    -0.0
# households              -0.1        0.1                -0.3          0.9             1.0         0.9         1.0            0.0              -0.0     0.1
# median_income           -0.1       -0.0                -0.1          0.2            -0.0         0.0         0.0            1.0               0.3     0.7
# rooms_per_person         0.1       -0.1                -0.1          0.1             0.1        -0.1        -0.0            0.3               1.0     0.2
# target                  -0.1       -0.0                 0.1          0.1             0.0        -0.0         0.1            0.7               0.2     1.0


def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
    
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear regression model.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for validation.
      
  Returns:
    A `LinearRegressor` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods

  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=construct_feature_columns(training_examples),
      optimizer=my_optimizer
  )
    
  # Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["median_house_value"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["median_house_value"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["median_house_value"], 
                                                    num_epochs=1, 
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period,
    )
    # Take a break and compute predictions.
    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")
  print("Final RMSE (on validation data):")
  print("%0.2f" % validation_root_mean_squared_error)
  
  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()
  plt.show()

  return linear_regressor

# Build a first model with a minimal set of features (median income and latitude)
minimal_features = [
  "median_income",
  "latitude",
]

minimal_training_examples = training_examples[minimal_features]
minimal_validation_examples = validation_examples[minimal_features]

# _ = train_model(
#     learning_rate=0.01,
#     steps=500,
#     batch_size=5,
#     training_examples=minimal_training_examples,
#     training_targets=training_targets,
#     validation_examples=minimal_validation_examples,
#     validation_targets=validation_targets)
# RMSE (on training data):
#   period 09 : 114.29

# Make Better Use of Latitude and Longitude
# Plotting latitude against house value
#plt.scatter(training_examples["latitude"], training_targets["median_house_value"])
# Result  => there isn't a linear relationship between those
# Instead => cluster of points with high values around LA and SF (latitude ~34 and ~38)
#plt.show()
#plt.scatter(training_examples["longitude"], training_targets["median_house_value"])
#plt.show()
# Same with longitude => -122.5 and -118
# Those spots correspond roughly to the most expensive cities in the state (SF and LA)

# Two better options to use this data

# 1) create new features distance_to_XX (here are 6 biggest cities in California)
# San Francisco SF => (37.8,-122.4)
# Los Angeles LA => (34.0, -118.2)
# San Diego SD => (32.7, -117.2)
# San Jose SJ => (37.3, -121.9)
# Fresno FR => (36.7, -119.8)
# Sacramento SA => (38.6, -121.5)

big_cities_coordinates = [
["SF", 37.8, -122.4],
["LA", 34.0, -118.2],
["SD", 32.7, -117.2],
["SJ", 37.3, -121.9],
["FR", 36.7, -119.8],
["SA", 38.6, -121.5]
]

print (big_cities_coordinates)
print (big_cities_coordinates[0])
print (big_cities_coordinates[0][1])

def select_and_transform_features(source_df):
  transformed_df = pd.DataFrame()
  # keep median income as is
  transformed_df["median_income"] = source_df["median_income"]
  for r in big_cities_coordinates:
    transformed_df["distance_to_%s" % r[0]] = ((source_df["latitude"] - r[1]).pow(2) + (source_df["longitude"] - r[2]).pow(2)).pow(0.5)
  return transformed_df

selected_training_examples = select_and_transform_features(training_examples)

selected_validation_examples = select_and_transform_features(validation_examples)

print (selected_training_examples.describe())
correlation_dataframe = selected_training_examples.copy()
correlation_dataframe["target"] = training_targets["median_house_value"]
print(correlation_dataframe.corr())
#                 median_income  distance_to_SF  distance_to_LA  distance_to_SD  distance_to_SJ  distance_to_FR  distance_to_SA  target
# median_income             1.0            -0.0            -0.1            -0.1            -0.0             0.0             0.0     0.7
# distance_to_SF           -0.0             1.0            -0.9            -0.9             1.0             0.5             1.0    -0.0
# distance_to_LA           -0.1            -0.9             1.0             1.0            -0.8            -0.2            -0.9    -0.1
# distance_to_SD           -0.1            -0.9             1.0             1.0            -0.9            -0.4            -0.9    -0.1
# distance_to_SJ           -0.0             1.0            -0.8            -0.9             1.0             0.6             1.0    -0.0
# distance_to_FR            0.0             0.5            -0.2            -0.4             0.6             1.0             0.6     0.1
# distance_to_SA            0.0             1.0            -0.9            -0.9             1.0             0.6             1.0     0.1
# target                    0.7            -0.0            -0.1            -0.1            -0.0             0.1             0.1     1.0

# no better correlation (going far from LA could get you closer to SF, after all)
# but maybe a linear combination of those would work

# _ = train_model(
#     learning_rate=0.02,
#     steps=1000,
#     batch_size=5,
#     training_examples=selected_training_examples,
#     training_targets=training_targets,
#     validation_examples=selected_validation_examples,
#     validation_targets=validation_targets)
# Final RMSE (on validation data):
# 91.09

# 2) Create bins for each square (lat_long_32_-122, lat_long_32_-121, ...)
# Works fine if the epicenters can't be located (big cities)
# creates more features (less efficient, but no one cares anymore)
# The course suggests to do this only for latitudes (so use bands instead of squares)

def select_and_transform_features_2(source_df):
  LATITUDE_RANGES = zip(range(32, 44), range(33, 45))
  selected_examples = pd.DataFrame()
  selected_examples["median_income"] = source_df["median_income"]
  for r in LATITUDE_RANGES:
    selected_examples["latitude_%d_to_%d" % r] = source_df["latitude"].apply(
      lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)
  return selected_examples

selected_training_examples = select_and_transform_features_2(training_examples)
selected_validation_examples = select_and_transform_features_2(validation_examples)

print (selected_training_examples.describe())
correlation_dataframe = selected_training_examples.copy()
correlation_dataframe["target"] = training_targets["median_house_value"]
print(correlation_dataframe.corr())
#                    median_income  latitude_32_to_33  latitude_33_to_34  latitude_34_to_35  latitude_35_to_36  ...  latitude_40_to_41  latitude_41_to_42  latitude_42_to_43  latitude_43_to_44  target
# target                       0.7               -0.1                0.1                0.1               -0.1  ...               -0.1               -0.1                nan                nan     1.0

# Correlation actually looks better (0.1 most of the time)
# Of course this might be an artefact of comparing a bool feature with a double feature. I don't know.

# _ = train_model(
#     learning_rate=0.02,
#     steps=1000,
#     batch_size=5,
#     training_examples=selected_training_examples,
#     training_targets=training_targets,
#     validation_examples=selected_validation_examples,
#     validation_targets=validation_targets)
# Final RMSE (on validation data):
# 83.35


# 3) Let's try creating bins for each square (lat_long_32_-122, lat_long_32_-121, ...)

def select_and_transform_features_3(source_df):
  LATITUDE_RANGES = range(32, 43)
  LONGITUDE_RANGES = range(-126, -115)
  selected_examples = pd.DataFrame()
  selected_examples["median_income"] = source_df["median_income"]
  for r in LATITUDE_RANGES:
    for s in LONGITUDE_RANGES:
      lats = source_df["latitude"].apply(
        lambda l: 1.0 if l >= r and l < r + 1 else 0.0)
      longs = source_df["longitude"].apply(
        lambda L: 1.0 if L >= s and L < s + 1 else 0.0)
      selected_examples["l_%d_L_%d" % (r,s)] = lats * longs
  return selected_examples

selected_training_examples = select_and_transform_features_3(training_examples)
print(selected_training_examples.describe())
selected_validation_examples = select_and_transform_features_3(validation_examples)



_ = train_model(
    learning_rate=0.02,
    steps=1000,
    batch_size=5,
    training_examples=selected_training_examples,
    training_targets=training_targets,
    validation_examples=selected_validation_examples,
    validation_targets=validation_targets)
# Final RMSE (on validation data):
# 81.58

# too slow and inefficient ... 11*11=121 new input features entering the model
# And all features have mostly 0s except for the rows actually in the right bin
# see feature-cross one-hot vectors

aaa = 1