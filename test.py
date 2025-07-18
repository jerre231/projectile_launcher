import lib
import numpy as np
import gui

def run_simulation(params):
    # Instanciando a classe
    sim = lib.SimuladorProjetil(**params)

    #sim.plotar_trajetoria_animada_comparada()
    sim.comparar_metodos()

if __name__ == "__main__":
    if gui.parameter_collect():
        collected_params = gui.params
        print("Parâmetros coletados com sucesso")
        run_simulation(params=collected_params)
    else:
        print("Coleta de parâmetros cancelada ou falha na entrada de dados.")
        print("A simulação não será iniciada.")