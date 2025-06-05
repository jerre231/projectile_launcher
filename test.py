import lib
import numpy as np

# Definir parâmetros físicos
params = {
    'massa': 1,             # kg (ex: bola de baseball)           # m² (ex: área frontal de baseball)
    'coeficiente_atrito': lib.calcularCoeficienteAtrito(0.5),         # Coeficiente de arrasto (esfera de 1 m de raio)
    'gravidade': 9.80665,    # m/s²
    'rotacao': np.pi/43200,      # kg/m³ (ao nível do mar)
    'tempo_de_forca': 1.5*(10**-3) # 1.5 ms já convertido em segundos
}
print(params['coeficiente_atrito'])

# Instanciando a classe
sim = lib.SimuladorProjetil(**params)

# Testando método RK4
angulo = 45
forca = 1000000
#sim.velocidadeInicial(forca, angulo) # TODO: quebra entre 340k e 350k Newtons de forca, porque?
#sim.comparar_dt([0.5,0.01,0.005,0.001])
#print(sim.plotar_trajetoria(0.001)) # Essa é a funcao para resolver o sistema

# Testando configuracoes
configuracoes = [
{'forca': 10000, 'angulo': 45, 'dt': 0.3},
{'forca': 10000, 'angulo': 45, 'dt': 0.2},
{'forca': 10000, 'angulo': 45, 'dt': 0.01},
{'forca': 10000, 'angulo': 45, 'dt': 0.001}]

sim.comparar_configuracoes(configuracoes)
