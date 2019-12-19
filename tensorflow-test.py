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
# Additionally, scale median_house_value to be in units of thousands, so it can be learned a little more easily with learning rates in a range that we usually use.
california_housing_dataframe["median_house_value"] /= 1000.0
print(california_housing_dataframe)
# [17000 rows x 9 columns]
print(california_housing_dataframe.describe())
#        longitude  latitude  housing_median_age  total_rooms  total_bedrooms  population  households  median_income  median_house_value
# count    17000.0   17000.0             17000.0      17000.0         17000.0     17000.0     17000.0        17000.0             17000.0
# mean      -119.6      35.6                28.6       2643.7           539.4      1429.6       501.2            3.9               207.3
# std          2.0       2.1                12.6       2179.9           421.5      1147.9       384.5            1.9               116.0
# min       -124.3      32.5                 1.0          2.0             1.0         3.0         1.0            0.5                15.0
# 25%       -121.8      33.9                18.0       1462.0           297.0       790.0       282.0            2.6               119.4
# 50%       -118.5      34.2                29.0       2127.0           434.0      1167.0       409.0            3.5               180.4
# 75%       -118.0      37.7                37.0       3151.2           648.2      1721.0       605.2            4.8               265.0
# max       -114.3      42.0                52.0      37937.0          6445.0     35682.0      6082.0           15.0               500.0


# In this exercise, we'll try to predict median_house_value, which will be our label (sometimes also called a target). We'll use total_rooms as our input feature.
# In order to import our training data into TensorFlow, we need to specify what type of data each feature contains. 
# There are two main types of data we'll use in this and future exercises:
#  - Categorical Data: Data that is textual. In this exercise, our housing data set does not contain any categorical 
#    features, but examples you might see would be the home style, the words in a real-estate ad.
#  - Numerical Data: Data that is a number (integer or float) and that you want to treat as a number. As we will discuss 
#    more later sometimes you might want to treat numerical data (e.g., a postal code) as if it were categorical.


# the Series is the data structure for a single column of a DataFrame, not only conceptually, but literally, 
# i.e. the data in a DataFrame is actually stored in memory as a collection of Series. 
print(type(california_housing_dataframe["total_rooms"]))
# <class 'pandas.core.series.Series'>
print(type(california_housing_dataframe[["total_rooms"]]))
# <class 'pandas.core.frame.DataFrame'>


# Define the input feature: total_rooms.
my_feature = california_housing_dataframe[["total_rooms"]]

# Configure a feature column for total_rooms. Use numeric_column, which specifies its data is numeric
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

# Define the label (our target, which is median_house_value.)
targets = california_housing_dataframe["median_house_value"]

# Next, we'll configure a linear regression model using LinearRegressor. We'll train this model using the GradientDescentOptimizer, 
# which implements Mini-Batch Stochastic Gradient Descent (SGD). The learning_rate argument controls the size of the gradient step.

# NOTE: To be safe, we also apply gradient clipping to our optimizer via clip_gradients_by_norm. Gradient clipping ensures 
# the magnitude of the gradients do not become too large during training, which can cause gradient descent to fail.


# Use gradient descent as the optimizer for training the model.
# TODO Module 'tensorflow_core._api.v2.train' has no 'GradientDescentOptimizer' memberpylint(no-member)
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001) #tf.optimizers.SGD(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configure the linear regression model with our feature columns and optimizer.
# Set a learning rate of 0.0000001 for Gradient Descent.
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    # NOTE: We'll continue to use this same input function in later exercises. 
    # For more detailed documentation of input functions and the Dataset API, see the TensorFlow Programmer's Guide :
    # https://www.tensorflow.org/guide/data (careful of tf v1/v2)

    # Convert pandas data (a DataFrame which is a "dict" of pandas Series) into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

# call train() on our linear_regressor to train the model.
_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature, targets),
    steps=100
)

# Let's make predictions on that training data, to see how well our model fit it during training.
# NOTE: Training error measures how well your model fits the training data, but it does not measure how well your model generalizes to new data. 
# In later exercises, you'll explore how to split your data to evaluate your model's ability to generalize.

# Create an input function for predictions.
# Note: Since we're making just one prediction for each example, we don't 
# need to repeat or shuffle the data here.
prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# Call predict() on the linear_regressor to make predictions.
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# Format predictions as a NumPy array, so we can calculate error metrics.
predictions = np.array([item['predictions'][0] for item in predictions])

# Print Mean Squared Error and Root Mean Squared Error.
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)
# Mean Squared Error (on training data): 56367.025
# Root Mean Squared Error (on training data): 237.417


# Let's compare the RMSE to the difference of the min and max of our targets:
min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)
# Min. Median House Value: 14.999
# Max. Median House Value: 500.001
# Difference between Min. and Max.: 485.002
# Our error spans nearly half the range of the target values. Can we do better?


calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
print(calibration_data.describe())
#        predictions  targets
# count      17000.0  17000.0
# mean           0.1    207.3
# std            0.1    116.0
# min            0.0     15.0
# 25%            0.1    119.4
# 50%            0.1    180.4
# 75%            0.2    265.0
# max            1.9    500.0

# It looks way off
# This was with a learning rate of 0.0000001 and 100 steps


# get a uniform random sample of the data so we can make a readable scatter plot.
sample = california_housing_dataframe.sample(n=300)

# Get the min and max total_rooms values.
x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

# Retrieve the final weight and bias generated during training.
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

# Get the predicted median_house_values for the min and max total_rooms values.
y_0 = weight * x_0 + bias 
y_1 = weight * x_1 + bias

# Plot our regression line from (x_0, y_0) to (x_1, y_1).
plt.plot([x_0, x_1], [y_0, y_1], c='r')

# Label the graph axes.
plt.ylabel("median_house_value")
plt.xlabel("total_rooms")

# Plot a scatter plot from our data sample.
plt.scatter(sample["total_rooms"], sample["median_house_value"])

# Display graph.
plt.show()
# After plotting, it's obvious that the learning rate is too low (or not enough steps)

aaa = 1