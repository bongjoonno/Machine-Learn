import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from math import log
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import TypeAlias
import random
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from tabpfn import TabPFNRegressor
from ucimlrepo import fetch_ucirepo