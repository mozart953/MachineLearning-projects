import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cols = ["fLength","fWidth", "fSize", "fConc",  "fConcl", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

df = pd.read_csv('./data/magic04.data', names=cols, encoding='utf-8')

df.head()