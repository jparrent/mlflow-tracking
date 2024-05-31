from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import argparse

cwd = Path(__file__).resolve()
print(cwd)
homedir = cwd.parent.parent
datadir = 'data/raw'
print(homedir / datadir)

df = pd.read_csv(homedir / datadir / 'chicago_traffic_data_slim.csv')
print(df.head)
print(df.columns)
print(df.info)

