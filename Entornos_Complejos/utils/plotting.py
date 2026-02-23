import matplotlib.pyplot as plt
import numpy as np


def plot_curve(
    values,
    title="Evolución",
    xlabel="Episodio",
    ylabel="Valor",
    moving_avg_window=None,
    label=None
):
    """
    Función genérica para dibujar una curva temporal.
    
    Parameters
    ----------
    values : list or np.array
        Serie temporal a representar.
    moving_avg_window : int or None
        Si se especifica, añade media móvil.
    """

    plt.figure(figsize=(6, 3))
    plt.plot(values, label=label if label else "Valor")

    if moving_avg_window is not None and moving_avg_window > 1:
        moving_avg = np.convolve(
            values,
            np.ones(moving_avg_window) / moving_avg_window,
            mode="valid"
        )
        plt.plot(
            range(moving_avg_window - 1, len(values)),
            moving_avg,
            linewidth=2,
            label=f"Media móvil ({moving_avg_window})"
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if label or moving_avg_window:
        plt.legend()

    plt.show()


def compare_algorithms(results_dict, title="Comparación de algoritmos", window=200):

    plt.figure(figsize=(8, 5))

    for name, values in results_dict.items():

        if window > 1:
            smoothed = np.convolve(
                values,
                np.ones(window)/window,
                mode='valid'
            )
            plt.plot(smoothed, label=name)
        else:
            plt.plot(values, label=name)

    plt.title(title)
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa por episodio")
    plt.legend()
    plt.grid(True)
    plt.show()

# def compare_algorithms(results_dict, title="Comparación de algoritmos"):
#     """
#     results_dict: dict
#         {"SARSA": lista_valores, "Q-Learning": lista_valores, ...}
#     """

#     plt.figure(figsize=(7, 4))

#     for name, values in results_dict.items():
#         plt.plot(values, label=name)

#     plt.title(title)
#     plt.xlabel("Episodio")
#     plt.ylabel("Recompensa promedio")
#     plt.legend()
#     plt.grid(True)
#     plt.show()