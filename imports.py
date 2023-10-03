import numpy as np
import random
import os
from pathlib import Path
import pymongo
import pickle
import datetime
import time
from datetime import *
from time import *
import gym
from gym import Env
# discrete is basically a set amount of actions (for us choosing from columns 0-6) that the env can do
from gym.spaces import Discrete
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
from keras import layers

