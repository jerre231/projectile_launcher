import lib
import numpy as np

# Definir parâmetros físicos
params = {
    'massa': 1,             # kg (ex: bola de baseball)           # m² (ex: área frontal de baseball)
    'coeficiente_atrito': 0.47,                 # Coeficiente de arrasto (esfera)
    'gravidade': 9.80665,    # m/s²
    'rotacao': np.pi/43200,      # kg/m³ (ao nível do mar)
}

# Definir o alvo e o ângulo
xf_objetivo = 100.0  # metros
angulo = np.deg2rad(45)  # Convertendo graus para radianos

# Encontrar velocidade inicial necessária
v0_encontrado = lib.ajustar_velocidade(xf_objetivo, angulo, params)

print(f"\nVelocidade inicial necessária: {v0_encontrado:.2f} m/s")

lib.plotar_trajetoria(v0_encontrado, angulo, params)