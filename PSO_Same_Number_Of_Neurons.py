# -*- coding: utf-8 -*-
"""PSO_KDD_TS.ipynb

author: Ishan FOOLELL & Kajal SOBHEE

"""

import copy
import numpy as np

class Particle(object):
    """Particle class for PSO

    This class encapsulates the behavior of each particle in PSO and provides
    an efficient way to do bookkeeping about the state of the swarm in any given
    iteration.

    Args:
        lower_bound (np.array): Vector of lower boundaries for particle dimensions.
        upper_bound (np.array): Vector of upper boundaries for particle dimensions.
        dimensions (int): Number of dimensions of the search space.
        objective function (function): Black-box function to evaluate.

    """
    def __init__(self,
                 lower_bound,
                 upper_bound,
                 dimensions,
                 objective_function):
        self.reset(dimensions, lower_bound, upper_bound, objective_function)

    def reset(self,
              dimensions,
              lower_bound,
              upper_bound,
              objective_function):
        """Particle reset

        Allows for reset of a particle without reallocation.

		Args:
			lower_bound (np.array): Vector of lower boundaries for particle dimensions.
			upper_bound (np.array): Vector of upper boundaries for particle dimensions.
			dimensions (int): Number of dimensions of the search space.

        """
        position = []
        for i in range(dimensions):
            if lower_bound[i] < upper_bound[i]:
                position.extend(np.random.randint(lower_bound[i], upper_bound[i] + 1, 1, dtype=int))
            elif lower_bound[i] == upper_bound[i]:
                position.extend(np.array([lower_bound[i]], dtype=int))
            else:
                assert False

        self.position = [position]

        self.velocity = [np.multiply(np.random.rand(dimensions),
                                     (upper_bound - lower_bound)).astype(int)]

        self.best_position = self.position[:]

        self.function_value = [objective_function(self.best_position[-1])]
        self.best_function_value = self.function_value[:]

    def update_velocity(self, omega, phip, phig, best_swarm_position):
        """Particle velocity update

		Args:
			omega (float): Velocity equation constant.
			phip (float): Velocity equation constant.
			phig (float): Velocity equation constant.
			best_swarm_position (np.array): Best particle position.

        """
        random_coefficient_p = np.random.uniform(size=np.asarray(self.position[-1]).shape)
        random_coefficient_g = np.random.uniform(size=np.asarray(self.position[-1]).shape)

        self.velocity.append(omega
                             * np.asarray(self.velocity[-1])
                             + phip
                             * random_coefficient_p
                             * (np.asarray(self.best_position[-1])
                                - np.asarray(self.position[-1]))
                             + phig
                             * random_coefficient_g
                             * (np.asarray(best_swarm_position)
                                - np.asarray(self.position[-1])))

        self.velocity[-1] = self.velocity[-1].astype(int)

    def update_position(self, lower_bound, upper_bound, objective_function):
        """Particle position update

		Args:
			lower_bound (np.array): Vector of lower boundaries for particle dimensions.
			upper_bound (np.array): Vector of upper boundaries for particle dimensions.
			objective function (function): Black-box function to evaluate.

        """
        new_position = self.position[-1] + self.velocity[-1]

        if np.array_equal(self.position[-1], new_position):
            self.function_value.append(self.function_value[-1])
        else:
            mark1 = new_position < lower_bound
            mark2 = new_position > upper_bound

            new_position[mark1] = lower_bound[mark1]
            new_position[mark2] = upper_bound[mark2]

            self.function_value.append(objective_function(self.position[-1]))

        self.position.append(new_position.tolist())

        if self.function_value[-1] < self.best_function_value[-1]:
            self.best_position.append(self.position[-1][:])
            self.best_function_value.append(self.function_value[-1])

class Pso(object):
    """PSO wrapper

    This class contains the particles and provides an abstraction to hold all the context
    of the PSO algorithm

    Args:
        swarmsize (int): Number of particles in the swarm
        maxiter (int): Maximum number of generations the swarm will run

    """
    def __init__(self, swarmsize=100, maxiter=100):
        self.max_generations = maxiter
        self.swarmsize = swarmsize

        self.omega = 0.5
        self.phip = 0.5
        self.phig = 0.5

        self.minstep = 1e-4
        self.minfunc = 1e-4

        self.best_position = [None]
        self.best_function_value = [1]

        self.particles = []

        self.retired_particles = []

    def run(self, function, lower_bound, upper_bound, kwargs=None):
        """Perform a particle swarm optimization (PSO)

		Args:
			objective_function (function): The function to be minimized.
			lower_bound (np.array): Vector of lower boundaries for particle dimensions.
			upper_bound (np.array): Vector of upper boundaries for particle dimensions.

		Returns:
			best_position (np.array): Best known position
			accuracy (float): Objective value at best_position
			:param kwargs:

        """
        if kwargs is None:
            kwargs = {}

        objective_function = lambda x: function(x, **kwargs)
        assert hasattr(function, '__call__'), 'Invalid function handle'

        assert len(lower_bound) == len(upper_bound), 'Invalid bounds length'

        lower_bound = np.array(lower_bound)
        upper_bound = np.array(upper_bound)

        assert np.all(upper_bound > lower_bound), 'Invalid boundary values'


        dimensions = len(lower_bound)

        self.particles = self.initialize_particles(lower_bound,
                                                   upper_bound,
                                                   dimensions,
                                                   objective_function)

        # Start evolution
        generation = 1
        while generation <= self.max_generations:
            for particle in self.particles:
                particle.update_velocity(self.omega, self.phip, self.phig, self.best_position[-1])
                particle.update_position(lower_bound, upper_bound, objective_function)

                if particle.best_function_value[-1] == 0:
                    self.retired_particles.append(copy.deepcopy(particle))
                    particle.reset(dimensions, lower_bound, upper_bound, objective_function)
                elif particle.best_function_value[-1] < self.best_function_value[-1]:
                    stepsize = np.sqrt(np.sum((np.asarray(self.best_position[-1])
                                               - np.asarray(particle.position[-1])) ** 2))

                    if np.abs(np.asarray(self.best_function_value[-1])
                              - np.asarray(particle.best_function_value[-1])) \
                            <= self.minfunc:
                        return particle.best_position[-1], particle.best_function_value[-1]
                    elif stepsize <= self.minstep:
                        return particle.best_position[-1], particle.best_function_value[-1]
                    else:
                        self.best_function_value.append(particle.best_function_value[-1])
                        self.best_position.append(particle.best_position[-1][:])



            generation += 1

        return self.best_position[-1], self.best_function_value[-1]

    def initialize_particles(self,
                             lower_bound,
                             upper_bound,
                             dimensions,
                             objective_function):
        """Initializes the particles for the swarm

		Args:
			objective_function (function): The function to be minimized.
			lower_bound (np.array): Vector of lower boundaries for particle dimensions.
			upper_bound (np.array): Vector of upper boundaries for particle dimensions.
			dimensions (int): Number of dimensions of the search space.

		Returns:
			particles (list): Collection or particles in the swarm

        """
        particles = []
        for _ in range(self.swarmsize):
            particles.append(Particle(lower_bound,
                                      upper_bound,
                                      dimensions,
                                      objective_function))
            if particles[-1].best_function_value[-1] < self.best_function_value[-1]:
                self.best_function_value.append(particles[-1].best_function_value[-1])
                self.best_position.append(particles[-1].best_position[-1])


        self.best_position = [self.best_position[-1]]
        self.best_function_value = [self.best_function_value[-1]]

        return particles

import pandas as pd
from tensorflow.keras.utils import get_file
from sklearn.model_selection import train_test_split

batch_size = 128
num_classes = 10
epochs = 100

try:
    path = get_file('kddcup.data_10_percent.gz', origin='http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz')
except:
    print('Error downloading')
    raise

df = pd.read_csv(path, header=None)
df.dropna(inplace=True,axis=1)

df.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome'
]

ENCODING = 'utf-8'

# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()
    if sd is None:
        sd = df[name].std()
    df[name] = (df[name] - mean) / sd
    
# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

# Now encode the feature vector

encode_numeric_zscore(df, 'duration')
encode_text_dummy(df, 'protocol_type')
encode_text_dummy(df, 'service')
encode_text_dummy(df, 'flag')
encode_numeric_zscore(df, 'src_bytes')
encode_numeric_zscore(df, 'dst_bytes')
encode_text_dummy(df, 'land')
encode_numeric_zscore(df, 'wrong_fragment')
encode_numeric_zscore(df, 'urgent')
encode_numeric_zscore(df, 'hot')
encode_numeric_zscore(df, 'num_failed_logins')
encode_text_dummy(df, 'logged_in')
encode_numeric_zscore(df, 'num_compromised')
encode_numeric_zscore(df, 'root_shell')
encode_numeric_zscore(df, 'su_attempted')
encode_numeric_zscore(df, 'num_root')
encode_numeric_zscore(df, 'num_file_creations')
encode_numeric_zscore(df, 'num_shells')
encode_numeric_zscore(df, 'num_access_files')
encode_numeric_zscore(df, 'num_outbound_cmds')
encode_text_dummy(df, 'is_host_login')
encode_text_dummy(df, 'is_guest_login')
encode_numeric_zscore(df, 'count')
encode_numeric_zscore(df, 'srv_count')
encode_numeric_zscore(df, 'serror_rate')
encode_numeric_zscore(df, 'srv_serror_rate')
encode_numeric_zscore(df, 'rerror_rate')
encode_numeric_zscore(df, 'srv_rerror_rate')
encode_numeric_zscore(df, 'same_srv_rate')
encode_numeric_zscore(df, 'diff_srv_rate')
encode_numeric_zscore(df, 'srv_diff_host_rate')
encode_numeric_zscore(df, 'dst_host_count')
encode_numeric_zscore(df, 'dst_host_srv_count')
encode_numeric_zscore(df, 'dst_host_same_srv_rate')
encode_numeric_zscore(df, 'dst_host_diff_srv_rate')
encode_numeric_zscore(df, 'dst_host_same_src_port_rate')
encode_numeric_zscore(df, 'dst_host_srv_diff_host_rate')
encode_numeric_zscore(df, 'dst_host_serror_rate')
encode_numeric_zscore(df, 'dst_host_srv_serror_rate')
encode_numeric_zscore(df, 'dst_host_rerror_rate')
encode_numeric_zscore(df, 'dst_host_srv_rerror_rate')

# display 5 rows

df.dropna(inplace=True,axis=1)
df[0:5]
# This is the numeric feature vector, as it goes to the neural net


# Convert to numpy - Classification
x_columns = df.columns.drop('outcome')
x = df[x_columns].values
dummies = pd.get_dummies(df['outcome']) # Classification
outcomes = dummies.columns
num_classes = len(outcomes)
y = dummies.values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import time

"""
# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
"""

def func(param):
  start_time = time.time()
  n,l,fk,dr = param[0],param[1],param[2],param[3]
 
  model = Sequential()
  model.add(Dense(10, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
  for i in range(l):
    if fk == 1:
      # Need input shape for first layer.
      model.add(Dense(n, activation='relu', input_dim=x_train.shape[1]))
    elif fk == 2:
      model.add(Dense(n, activation='sigmoid', input_dim=x_train.shape[1]))

  model.add(Dropout(dr/10))  # hard-coded dropout
  model.add(Dense(1, kernel_initializer='normal'))
  model.add(Dense(y.shape[1],activation='softmax'))
  model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
  
  cp = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')];

  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(x_test, y_test),callbacks=cp)
  
  score = model.evaluate(x_test, y_test, verbose=0)

  # loss, val
  time_taken = time.time() - start_time
  print('current config:',param,'val:',score[1],' duration: ', time_taken)
  return score[1]

##################################################################
pso = Pso(swarmsize=4,maxiter=14)

bp,value = pso.run(func,[1,1,1,1],[16,16,2,10])

v = func(bp);

##################################################################

print('Test loss:', bp)
print('Test accuracy:', value,v)