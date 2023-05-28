import pandas as pd

def house_loader(datasetPath):
    df =pd.read_csv(datasetPath)
    # print(df.info())
    return df

# house_loader("F:\Master\Deep learning\HW\datasets\ParisHousing/ParisHousing.csv")