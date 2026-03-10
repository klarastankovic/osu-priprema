import pandas as pd
import numpy as np
data = {'country': ['Italy', 'Spain', 'Greece', 'France', 'Portugal'],
        'population': [59, 47, 11, 68, 10],
        'code': [39, 34, 30, 33, 351]}
countries = pd.DataFrame(data, columns=['country', 'population', 'code'])
print(countries)