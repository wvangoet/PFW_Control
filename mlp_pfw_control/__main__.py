
import numpy as np
import pandas as pd

def rescale_column(df,key):
    max = np.max(df[key])
    min = np.min(df[key])
    df[key] = (df[key] - min) / (max - min)
    return min,max

def unscale_data(key,scale_dict,value):
    return value*(scale_dict[key][1]-scale_dict[key][0])+scale_dict[key][0]

def rescale_data(df):
    scale_dict={}
    for key in df.keys():
        min,max = rescale_column(df,key)
        scale_dict[key]=[min,max]
    return scale_dict

def rescale_datapoint(key,scale_dict,value):
    return (value - scale_dict[key][0])/(scale_dict[key][1]-scale_dict[key][0])


