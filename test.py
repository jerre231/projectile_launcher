import lib
import numpy as np
import gui

def run_simulation(params):
    # Instanciando a classe
    sim = lib.SimuladorProjetil(**params)

    #sim.plotar_trajetoria_animada_comparada()
    sim.comparar_metodos()

def sim_comparison(params):
    sim = lib.SimuladorProjetil(**params)

    # Testando configuracoes
    configuracoes = [
    {'forca': 10000, 'angulo': 45, 'dt': 0.3},
    {'forca': 10000, 'angulo': 45, 'dt': 0.2},
    {'forca': 10000, 'angulo': 45, 'dt': 0.01},
    {'forca': 10000, 'angulo': 45, 'dt': 0.001}]

    sim.comparar_configuracoes(configuracoes)
    return True

if __name__ == "__main__":
    if gui.parameter_collect():
        collected_params = gui.params
        print("Parâmetros coletados com sucesso")
        run_simulation(params=collected_params)
    else:
        print("Coleta de parâmetros cancelada ou falha na entrada de dados.")
        print("A simulação não será iniciada.")