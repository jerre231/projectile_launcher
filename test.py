import lib
import numpy as np

# Definir parâmetros físicos
params = {
    'massa': 1,             # kg (ex: bola de baseball)           # m² (ex: área frontal de baseball)
    'coeficiente_atrito': 0.47,                 # Coeficiente de arrasto (esfera)
    'gravidade': 9.80665,    # m/s²
    'rotacao': np.pi/43200,      # kg/m³ (ao nível do mar)
}

# Instanciando a classe
sim = lib.SimuladorProjetil(**params)

# Ajustar velocidade
angulo_rad = np.radians(45)
xf = 100
v0 = sim.ajustar_velocidade(xf, angulo_rad)

# Plotar a trajetória
sim.plotar_trajetoria(v0, angulo_rad)