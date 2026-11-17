import numpy as np
from utils.projection import project_simplex

class ProjectedGradientDescent:
    def __init__(self, gradient_func, eta=0.01, tol=1e-8, max_iter=1000):
        self.gradient_func = gradient_func
        self.eta = eta
        self.tol = tol
        self.max_iter = max_iter

    def optimize(self, w0):
        w = w0.copy()
        history_w = [w.copy()]
        history_dist = []
        
        for i in range(self.max_iter):
            grad = self.gradient_func(w)
            w_step = w - self.eta * grad
            w_new = project_simplex(w_step)
            
            history_w.append(w_new.copy())
            history_dist.append(np.linalg.norm(w_new - w))
            
            if history_dist[-1] < self.tol:
                break
            w = w_new
            
        return w, history_w, history_dist
