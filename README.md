# EML - Aprendizaje por Refuerzo
## Información

- **Alumnos:**  
Montesinos Bravo, Jorge  
Oliva Marín, Manuel  
Ouardi Bennane, Mariam  

- **Asignatura:** Extensiones de Machine Learning
- **Profesor**: Hernández Molinero, Luis Daniel
- **Curso:** 2025/2026
- **Grupo:** *MontesinosOlivaOuardi*


## Descripción
<!-- [Breve descripción del trabajo y sus objetivos] -->
Este repositorio contiene el desarrollo completo de la práctica de Aprendizaje por Refuerzo. El trabajo se divide en dos grandes bloques: una primera parte centrada en el problema del bandido de k-brazos y una segunda parte dedicada al estudio de entornos más complejos, incluyendo métodos tabulares y control con aproximación de funciones.

El objetivo principal es analizar, implementar y comparar distintos algoritmos de aprendizaje por refuerzo en entornos discretos y continuos, evaluando su comportamiento y rendimiento experimental.


## Estructura
<!-- [Explicación de la organización del repositorio] -->
```
/
├── k_brazos/
├── Entornos_Complejos/
├── docs/
├── requirements.txt
└── README.md
```

- `k_brazos/`: directorio correspondiente a la parte 1 de la práctica (problema del bandido de k-brazos).
- `Entornos_Complejos/`: directorio correspondiente a la parte 2 de la práctica (métodos tabulares y control con aproximaciones).
- `docs/` : documentación de la práctica en formato PDF (`informe.pdf`) y archivos `.tex` comprimidos en un zip (`EML_Informe.zip`).
- `requirements.txt`: lista de dependencias necesarias para ejecutar el trabajo (en local).
- `README.md`: este fichero.


Cada carpeta contiene su propio fichero `README.md` con una descripción detallada de su organización interna.


## Instalación y uso
<!-- [Instrucciones si son necesarias] -->

El proyecto ha sido desarrollado en Python 3.13.2 y ejecutado en Google Colab.

Para instalar las dependencias (en caso de ejecutar en local):

```bash
pip install -r requirements.txt
```

Los experimentos pueden ejecutarse directamente desde los notebooks `.ipynb`.

## Tecnologías
<!-- [Lista de lenguajes, frameworks, etc.] -->

- Python 3.13.2
- NumPy
- Matplotlib
- Gymnasium
- PyTorch
- Jupyter Notebook / Google Colab

