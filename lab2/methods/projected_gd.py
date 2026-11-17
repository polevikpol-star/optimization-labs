import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.projection import project_simplex

class ProjectedGradientDescent:
    def __init__(self, oracle, eta=None, tol=1e-8, max_iter=1000):
        self.oracle = oracle
        self.tol = tol
        self.max_iter = max_iter
        
        # Берем шаг 1/L из матрицы A (это ковариационная матрица внутри оракула)
        if eta is None:
            L = np.max(np.linalg.eigvalsh(oracle.A))
            self.eta = 1.0 / L
        else:
            self.eta = eta

    def optimize(self, w0):
        w = w0.copy()
        history_w = [w.copy()]
        history_dist = []
        
        for i in range(self.max_iter):
            # Обращаемся к градиенту внутри оракула
            grad = self.oracle.grad(w) 
            w_step = w - self.eta * grad
            w_new = project_simplex(w_step)
            
            history_w.append(w_new.copy())
            history_dist.append(np.linalg.norm(w_new - w))
            
            if history_dist[-1] < self.tol:
                print(f"Сошлось за {i} итераций!")
                break
            w = w_new
            
        return w, history_w, history_dist
