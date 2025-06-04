import lib
import numpy as np

# Definir parâmetros físicos
params = {
    'massa': 1,             # kg (ex: bola de baseball)           # m² (ex: área frontal de baseball)
    'coeficiente_atrito': 0.47,                 # Coeficiente de arrasto (esfera)
    'gravidade': 9.80665,    # m/s²
    'rotacao': np.pi/43200,      # kg/m³ (ao nível do mar)
    'tempo_de_forca': 1.5*(10**-3) # 1.5 ms já convertido em segundos
}

# Instanciando a classe
sim = lib.SimuladorProjetil(**params)

# Testando método RK4
sim.velocidadeInicial(340000, 45) # TODO: quebra entre 340k e 350k Newtons de forca, porque?
print(sim.plotar_trajetoria()) # Essa é a funcao para resolver o sistema

# Ajustar velocidade
#angulo_rad = np.radians(45)
#xf = 15
#v0 = sim.ajustar_velocidade(xf, angulo_rad)

# Plotar a trajetória
#sim.plotar_trajetoria(v0, angulo_rad)