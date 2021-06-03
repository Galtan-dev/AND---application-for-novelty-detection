import pandas as pd
import numpy as np
import matplotlib as plt
import padasip

hodnoty = open("Bankrot.csv", "r")
data = {
    "data": hodnoty
}
array = np.array(pd.DataFrame(data))

A = []
B = []
for j in range(0,6189,1):
    B.insert(0,array[j])
for i in range(0,6189,6):
    A.insert(0,array[i])
    np.delete(B, i)
print(A,B)