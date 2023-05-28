import pandas as pd
import numpy as np

def preprocess(df, input_type):
    if input_type == 'Normalized':
        df = df.apply(lambda iterator: ((iterator - iterator.min())/(iterator.max() - iterator.min())))
        target = df['price']
        target = target.to_numpy()
        df.drop(columns=['price'], inplace=True)
        X_ = df.to_numpy()
    elif input_type == 'Standardized':
        df = df.apply(lambda iterator: ((iterator - iterator.mean())/iterator.std()))
        target = df['price']
        target = target.to_numpy()
        df.drop(columns=['price'], inplace=True)
        X_ = df.to_numpy()
    return X_, target

