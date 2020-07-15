# -*- coding: utf-8 -*-
"""
Detect Outliners

@author: Varshtih
"""
from pyod.models.iforest import IForest
import numpy as np

def detect_outliner(x_train_Imp_En):
    """
    

    Parameters
    ----------
    x_train_Imp_En : TYPE
        DESCRIPTION.

    Returns
    -------
    x_train_outliner_data : TYPE
        DESCRIPTION.

    """
    outliners = IForest(contamination=0.05)
    x_train_outliner_data = x_train_Imp_En.copy()
    outliners.fit(x_train_outliner_data)
    x_train_outliner_anamoly_score = outliners.decision_function(x_train_outliner_data)*-1
    x_train_outliner_detection = outliners.predict(x_train_outliner_data)
    x_train_outliner_data['Outliner'] = x_train_outliner_detection.tolist()
    x_train_outliner_count = np.count_nonzero(x_train_outliner_detection == 1)
    
    return x_train_outliner_data
