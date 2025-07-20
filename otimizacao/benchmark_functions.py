import numpy as np

def ackley(vector:np.ndarray, dimensions:int=None, a:float=20, b:float=0.2, c:float=2*np.pi):
    if dimensions is not None and dimensions == vector.shape[0]:
        raise Exception("Number of dimensions is different from the vector shape")
    d = vector.shape[0]
    
    return -a * np.exp(-b*np.sqrt((1/d)*np.sum(vector**2))) - np.exp((1/d)*np.sum(np.cos(c*vector))) + a + np.exp(1)

def rastrigin(vector:np.ndarray, dimensions:int=None):
    if dimensions is not None and dimensions == vector.shape[0]:
        raise Exception("Number of dimensions is different from the vector shape")
    d = vector.shape[0]
    
    return 10*d + np.sum(vector**2 - 10*np.cos(2*np.pi*vector))


def schwefel(vector:np.ndarray, dimensions:int=None):
    if dimensions is not None and dimensions == vector.shape[0]:
        raise Exception("Number of dimensions is different from the vector shape")
    d = vector.shape[0]
    
    return 418.9829*d - np.sum(vector*np.sin(np.sqrt(np.abs(vector))))

def rosenbrock(vector:np.ndarray, dimensions:int=None, a:float=20, b:float=0.2, c:float=2*np.pi):
    if dimensions is not None and dimensions == vector.shape[0]:
        raise Exception("Number of dimensions is different from the vector shape")
    d = vector.shape[0]
    somatorio = 0
    for i in range(d-1):
        somatorio += 100 * (vector[i+1] - vector[i]**2)**2 + (vector[i] - 1)**2
    return somatorio
            
print(rastrigin(np.array([0,0,0,0,0,0,0])))