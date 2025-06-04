import numpy as np
import matplotlib.pyplot as plt

class SimuladorProjetil:
    def __init__(self, massa, coeficiente_atrito, gravidade, rotacao, tempo_de_forca):
        """
        Inicializa o simulador com os parâmetros físicos.
        """
        self.m = massa
        self.k = coeficiente_atrito
        self.g = gravidade
        self.tF = tempo_de_forca
        self.w = rotacao  # não usado ainda, mas incluído como atributo

        self.angulo = 0 # a ser definido pelo método velocidadeInicial
        self.forca = 0 # a ser definido pelo método velocidadeInicial
        self.estado_inicial = [] # [x0, y0, vx0, vy0]

    def velocidadeInicial(self, forca, angulo):
        """
        Primeiro método a ser chamado, recebe os parametros iniciais e define v0.
        """
        vx0 = (forca * self.tF * np.cos(np.radians(angulo))) / self.m
        vy0 = (forca * self.tF * np.sin(np.radians(angulo))) / self.m

        self.forca = forca
        self.angulo = angulo
        self.estado_inicial = [0, 0, vx0, vy0]

        return self.estado_inicial

    def EDOs(self, estado):
        """
        Define o sistema de EDOs para o projétil com arrasto. Método a ser usado por outros métodos
        """
        vx = estado[2]
        vy = estado[3]

        Fd = np.sqrt(vx**2 + vy**2)

        # Força de arrasto
        Fd_x = self.k * Fd * vx
        Fd_y = self.k * Fd * vy

        # Equações diferenciais
        dxdt = vx
        dydt = vy
        dvxdt = -Fd_x / self.m
        dvydt = (-Fd_y / self.m) - self.g

        return np.array([dxdt, dydt, dvxdt, dvydt])

    def runge_kutta4(self, estado, dt):
        """
        Executa um passo de Runge-Kutta 4ª ordem. Método a ser utilizado por outros métodos.
        """
        k1 = self.EDOs(estado)
        k2 = self.EDOs(estado + 0.5 * dt * k1)
        k3 = self.EDOs(estado + 0.5 * dt * k2)
        k4 = self.EDOs(estado + dt * k3)

        return estado + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def ajustar_velocidade(self, xf_objetivo, angulo, vmin=1.0, vmax=300.0, tolerancia=1e-2, max_iter=100): # TODO: Funcao incorreta para definir v0, descobrir como definir pelo RK4 ou leis da física
        """
        Ajusta a velocidade inicial para atingir o alvo usando o método da bisseção.
        """
        for i in range(max_iter):
            v0 = (vmin + vmax) / 2
            erro = self.simular_impacto(v0, angulo, xf_objetivo)
            
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

    def plotar_trajetoria(self, dt=0.01):
        """
        Simula e plota a trajetória do projétil.
        """
        estado = self.estado_inicial
        xs = [estado[0]] # x0
        ys = [estado[1]] # y0
        v0 = np.sqrt(estado[2]**2 + estado[3]**2)

        while estado[1] >= 0: # enquanto y não for negativo, ou seja, para quando tocar o chão novamente
            estado = self.runge_kutta4(estado, dt)
            xs.append(estado[0])
            ys.append(estado[1])

        print(v0)
        print(self.angulo)

        plt.figure(figsize=(10, 6))
        plt.plot(xs, ys, label=f"v0 = {v0:.2f} m/s, ângulo = {self.angulo:.1f}°")
        plt.xlabel("Distância horizontal (m)")
        plt.ylabel("Altura (m)")
        plt.title("Trajetória do projétil com arrasto")
        plt.grid(True)
        plt.legend()
        plt.show()
