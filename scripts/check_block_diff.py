import pandas as pd
import numpy as np

scores = pd.read_csv('results/features/sheffield_nai_scores.csv')

# Порівняти D_D vs інші блоки для ASD і CTRL
for block in ['D_SE', 'D_C', 'D_G', 'D_D']:
    asd = scores[scores['group']=='ASD'][block]
    ctrl = scores[scores['group']=='CTRL'][block]
    print(f'{block}: ASD={asd.mean():.3f}±{asd.std():.3f}  CTRL={ctrl.mean():.3f}±{ctrl.std():.3f}  diff={asd.mean()-ctrl.mean():.3f}')
