import numpy as np
import matplotlib.pyplot as plt

'''
Predefinido:
m -> massa do projétil em kg
k -> coeficiente de arrasto
g -> constante gravitacional
w -> velocidade angular de rotação do planeta (rad/s)

Definido pelo usuário:
o -> ângulo
xf -> local de queda

Resultado:
f -> força do canhão
vo -> velocidade inicial (derivado da força do canhão)
'''

def EDOs(estado, params):
    

    """
    Função que define o sistema de EDOs para o projétil com arrasto.

    Parâmetros:
    t : float
        Tempo atual (não usado diretamente, mas necessário para o integrador).
    estado : array_like
        Vetor de estado [x, y, vx, vy].
    params : dict
        Dicionário com parâmetros físicos (massa, coef atrito, gravidade, rotação terra).

    Retorna:
    derivadas : ndarray
        Derivadas [dx/dt, dy/dt, dvx/dt, dvy/dt].
    """
    x, y, vx, vy = estado
    m = params['massa']
    k = params['coeficiente_atrito']
    g = params['gravidade']
    w = params['rotacao']

    v = np.sqrt(vx**2 + vy**2) # módulo da velocidade

    # Força de arrasto
    Fd_x = k * vx
    Fd_y = k * vy

    # Equações diferenciais
    dxdt = vx
    dydt = vy
    dvxdt = Fd_x / m
    dvydt = (Fd_y / m) - g

    return np.array([dxdt, dydt, dvxdt, dvydt])

def runge_kutta4(estado, dt, params):
    """
    Executa um passo de Runge-Kutta 4ª ordem para as EDOs do projétil.

    estado : array_like
        Estado atual [x, y, vx, vy].
    dt : float
        Passo de tempo.
    params : dict
        Parâmetros físicos.

    Retorna:
    novo_estado : ndarray
        Estado atualizado após dt.
    """
    k1 = EDOs(estado, params)
    k2 = EDOs(estado + 0.5 * dt * k1, params)
    k3 = EDOs(estado + 0.5 * dt * k2, params)
    k4 = EDOs(estado + dt * k3, params)

    novo_estado = estado + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    return novo_estado

def simular_impacto(v0, angulo, params, xf_objetivo, dt=0.01):
    """
    Simula a trajetória usando Runge-Kutta 4 e calcula o erro até o alvo.
    
    v0 : float
        Velocidade inicial (módulo).
    angulo : float
        Ângulo de disparo (radianos).
    params : dict
        Parâmetros físicos.
    xf_objetivo : float
        Posição horizontal final desejada.
    dt : float
        Passo de tempo para a simulação.

    Retorna:
    erro : float
        Diferença entre posição de impacto e alvo.
    """
    # Condições iniciais
    x0, y0 = 0.0, 0.0
    vx0 = v0 * np.cos(angulo)
    vy0 = v0 * np.sin(angulo)
    estado = np.array([x0, y0, vx0, vy0])
    
    while estado[1] >= 0:  # Enquanto não tocar o chão
        estado = runge_kutta4(estado, dt, params)
    
    x_final = estado[0]
    erro = x_final - xf_objetivo
    return erro

def ajustar_velocidade(xf_objetivo, angulo, params, vmin=1.0, vmax=300.0, tolerancia=1e-2, max_iter=100):
    """
    Ajusta a velocidade inicial para atingir xf_objetivo usando bisseção.
    
    xf_objetivo : float
        Posição horizontal desejada.
    angulo : float
        Ângulo de disparo (radianos).
    params : dict
        Parâmetros físicos.
    vmin : float
        Velocidade mínima inicial para busca.
    vmax : float
        Velocidade máxima inicial para busca.
    tolerancia : float
        Tolerância de erro em metros.
    max_iter : int
        Número máximo de iterações.

    Retorna:
    v0_final : float
        Velocidade inicial encontrada.
    """
    for i in range(max_iter):
        v0 = (vmin + vmax) / 2
        erro = simular_impacto(v0, angulo, params, xf_objetivo)
        
        print(f"Iteração {i}: v0 = {v0:.4f} m/s, erro = {erro:.4f} m")
        
        if abs(erro) < tolerancia:
            print(f"Convergiu em {i} iterações!")
            return v0
        
        if erro > 0:
            vmax = v0
        else:
            vmin = v0
    
    print("Aviso: número máximo de iterações atingido.")
    return (vmin + vmax) / 2

def plotar_trajetoria(v0, angulo, params, dt=0.01):
    """
    Simula e plota a trajetória do projétil.

    v0 : float
        Velocidade inicial.
    angulo : float
        Ângulo de disparo (radianos).
    params : dict
        Parâmetros físicos.
    dt : float
        Passo de tempo.
    """
    # Condições iniciais
    x0, y0 = 0.0, 0.0
    vx0 = v0 * np.cos(angulo)
    vy0 = v0 * np.sin(angulo)
    estado = np.array([x0, y0, vx0, vy0])

    # Listas para armazenar os resultados
    xs = [x0]
    ys = [y0]

    while estado[1] >= 0:
        estado = runge_kutta4(estado, dt, params)
        xs.append(estado[0])
        ys.append(estado[1])

    # Plotando
    plt.figure(figsize=(10,6))
    plt.plot(xs, ys, label=f"v0 = {v0:.2f} m/s, ângulo = {np.degrees(angulo):.1f}°")
    plt.xlabel("Distância horizontal (m)")
    plt.ylabel("Altura (m)")
    plt.title("Trajetória do projétil com arrasto")
    plt.grid(True)
    plt.legend()
    plt.show()