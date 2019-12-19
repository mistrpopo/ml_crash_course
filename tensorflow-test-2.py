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



# wrap all the code from the first test in a single function for convenience. 

def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
  """Trains a linear regression model of one feature.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    input_feature: A `string` specifying a column from `california_housing_dataframe`
      to use as input feature.
  """
  
  periods = 10
  steps_per_period = steps / periods

  my_feature = input_feature
  my_feature_data = california_housing_dataframe[[my_feature]]
  my_label = "median_house_value"
  targets = california_housing_dataframe[my_label]

  # Create feature columns.
  feature_columns = [tf.feature_column.numeric_column(my_feature)]
  
  # Create input functions.
  training_input_fn = lambda:my_input_fn(my_feature_data, targets, batch_size=batch_size)
  prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)
  
  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )

  # Set up to plot the state of our model's line each period.
  plt.figure(figsize=(15, 6))
  plt.subplot(1, 2, 1)
  plt.title("Learned Line by Period")
  plt.ylabel(my_label)
  plt.xlabel(my_feature)
  sample = california_housing_dataframe.sample(n=300)
  plt.scatter(sample[my_feature], sample[my_label])
  colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  root_mean_squared_errors = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    predictions = linear_regressor.predict(input_fn=prediction_input_fn)
    predictions = np.array([item['predictions'][0] for item in predictions])
    
    # Compute loss.
    root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(predictions, targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    root_mean_squared_errors.append(root_mean_squared_error)
    # Finally, track the weights and biases over time.
    # Apply some math to ensure that the data and line are plotted neatly.
    y_extents = np.array([0, sample[my_label].max()])
    
    weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

    x_extents = (y_extents - bias) / weight
    x_extents = np.maximum(np.minimum(x_extents,
                                      sample[my_feature].max()),
                           sample[my_feature].min())
    y_extents = weight * x_extents + bias
    plt.plot(x_extents, y_extents, color=colors[period]) 
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.subplot(1, 2, 2)
  plt.ylabel('RMSE')
  plt.xlabel('Periods')
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(root_mean_squared_errors)
  plt.show()

  # Output a table with calibration data.
  calibration_data = pd.DataFrame()
  calibration_data["predictions"] = pd.Series(predictions)
  calibration_data["targets"] = pd.Series(targets)
  display.display(calibration_data.describe())

  print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)


# train_model(
#     learning_rate=0.001,
#     steps=10,
#     batch_size=1
# )
#        predictions  targets
# count      17000.0  17000.0
# mean         105.7    207.3
# std           87.2    116.0
# min            0.1     15.0
# 25%           58.5    119.4
# 50%           85.1    180.4
# 75%          126.0    265.0
# max         1517.4    500.0
# Final RMSE : 169.47


# 10x smaller learning rate, 10x more steps
# progress starts similarly, but when reaching close to the minimum, the RMSE on the first set of parameters went back up and down again.
# For this one, it kept going down.
# train_model(
#     learning_rate=0.0001,
#     steps=100,
#     batch_size=5
# )
#        predictions  targets
# count      17000.0  17000.0
# mean         108.4    207.3
# std           89.4    116.0
# min            0.1     15.0
# 25%           59.9    119.4
# 50%           87.2    180.4
# 75%          129.2    265.0
# max         1555.4    500.0
# Final RMSE (on training data): 168.84


# Add more steps since the RMSE kept going down in the previous example
train_model(
    learning_rate=0.0001,
    steps=400,
    batch_size=5
)
# Training model...
# RMSE (on training data):
#   period 00 : 194.62
#   period 01 : 171.74
#   period 02 : 166.39
#   period 03 : 168.28
#   period 04 : 173.57
#   period 05 : 179.23
#   period 06 : 177.98
#   period 07 : 184.78
#   period 08 : 179.23
#   period 09 : 174.58
# Model training finished.
#        predictions  targets
# count      17000.0  17000.0
# mean         171.8    207.3
# std          141.7    116.0
# min            0.1     15.0
# 25%           95.0    119.4
# 50%          138.2    180.4
# 75%          204.8    265.0
# max         2465.7    500.0
# Final RMSE (on training data): 174.58

# So, that's interesting, since it oscillated quite a bit after the minimum was reached very quickly (period 2), and still didn't reach a minimum.

aaa = 1