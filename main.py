import pandas as pd
import numpy as np

hodnoty = open("Bankrot.csv","r")
data = {
    "data": hodnoty
}

array = np.array(pd.DataFrame(data))
print(array)