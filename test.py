import lib
import numpy as np
import gui

"""
# Definir parâmetros físicos
params = {
    'massa': 1,             # kg (ex: bola de baseball)           # m² (ex: área frontal de baseball)
    'coeficiente_atrito': lib.calcularCoeficienteAtrito(0.5),         # Coeficiente de arrasto (esfera de 1 m de raio)
    'gravidade': 9.80665,    # m/s²
    'rotacao': np.pi/43200,      # kg/m³ (ao nível do mar)
    'tempo_de_forca': 1.5*(10**-3) # 1.5 ms já convertido em segundos
}
print(params['coeficiente_atrito'])
"""
def run_simulation(params):
    # Instanciando a classe
    sim = lib.SimuladorProjetil(**params)
    #sim.plotar_trajetoria()
    sim.plotar_trajetoria_animada()

    # TODO: quebra entre 340k e 350k Newtons de forca, porque?

def sim_comparison(params):   # TODO: Definir esta função
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