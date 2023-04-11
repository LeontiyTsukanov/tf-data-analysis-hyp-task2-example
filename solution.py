import pandas as pd
import numpy as np
from hyppo.ksample import MMD

chat_id = 407415686 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    flag = False
    if MMD(compute_kernel = "rbf", gamma = 0.7).test(x, y)[1] >= 0.03:
        flag = True
    return flag # Ваш ответ, True или False
