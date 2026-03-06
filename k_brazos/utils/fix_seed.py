import os
import gc
import torch
import numpy as np

def fix_seed(seed=100):
    # Liberación de memoria para evitar problemas de consumo en GPU
    gc.collect() # Ejecuta el recolector de basura de Python
    torch.cuda.empty_cache() # Vacía la caché de memoria en GPU

    # Depuración de errores en CUDA
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # Muestra errores de CUDA en el punto exacto donde ocurren
    # Configuración de la semilla para reproducibilidad

    # Fijar la semilla en NumPy
    np.random.seed(seed) # Para generar números aleatorios consistentes en NumPy
    np.random.default_rng(seed) # Establece una instancia del generador de NumPy con la misma semilla

    # Fijar la semilla en Python
    os.environ['PYTHONHASHSEED'] = str(seed) # Evita variabilidad en hashing de Python

    # Fijar la semilla en PyTorch
    torch.manual_seed(seed) # Asegura resultados reproducibles en operaciones de PyTorch
    if torch.cuda.is_available(): # Si hay GPU disponible
        torch.cuda.manual_seed(seed) # Fija la semilla para la GPU
        torch.backends.cudnn.deterministic = True # Hace las operaciones de CUDNN determinísticas
        torch.backends.cudnn.benchmark = False # Desactiva optimizaciones de CUDNN para evitar variabilidad
