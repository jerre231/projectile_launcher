import numpy as np
import matplotlib.pyplot as plt

class SimuladorProjetil:
    def __init__(self, massa, coeficiente_atrito, gravidade, rotacao):
        """
        Inicializa o simulador com os parâmetros físicos.
        """
        self.m = massa
        self.k = coeficiente_atrito
        self.g = gravidade
        self.w = rotacao  # não usado ainda, mas incluído como atributo

    def EDOs(self, estado):
        """
        Define o sistema de EDOs para o projétil com arrasto.
        """
        x, y, vx, vy = estado
        v = np.sqrt(vx**2 + vy**2)

        # Força de arrasto
        Fd_x = self.k * vx
        Fd_y = self.k * vy

        # Equações diferenciais
        dxdt = vx
        dydt = vy
        dvxdt = -Fd_x / self.m
        dvydt = (-Fd_y / self.m) - self.g

        return np.array([dxdt, dydt, dvxdt, dvydt])

    def runge_kutta4(self, estado, dt):
        """
        Executa um passo de Runge-Kutta 4ª ordem.
        """
        k1 = self.EDOs(estado)
        k2 = self.EDOs(estado + 0.5 * dt * k1)
        k3 = self.EDOs(estado + 0.5 * dt * k2)
        k4 = self.EDOs(estado + dt * k3)

        return estado + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def simular_impacto(self, v0, angulo, xf_objetivo, dt=0.01):
        """
        Simula a trajetória e calcula o erro em relação ao alvo.
        """
        x0, y0 = 0.0, 0.0
        vx0 = v0 * np.cos(angulo)
        vy0 = v0 * np.sin(angulo)
        estado = np.array([x0, y0, vx0, vy0])

        while estado[1] >= 0:
            estado = self.runge_kutta4(estado, dt)

        erro = estado[0] - xf_objetivo
        return erro

    def ajustar_velocidade(self, xf_objetivo, angulo, vmin=1.0, vmax=300.0, tolerancia=1e-2, max_iter=100):
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

    def plotar_trajetoria(self, v0, angulo, dt=0.01):
        """
        Simula e plota a trajetória do projétil.
        """
        x0, y0 = 0.0, 0.0
        vx0 = v0 * np.cos(angulo)
        vy0 = v0 * np.sin(angulo)
        estado = np.array([x0, y0, vx0, vy0])

        xs = [x0]
        ys = [y0]

        while estado[1] >= 0:
            estado = self.runge_kutta4(estado, dt)
            xs.append(estado[0])
            ys.append(estado[1])

        plt.figure(figsize=(10, 6))
        plt.plot(xs, ys, label=f"v0 = {v0:.2f} m/s, ângulo = {np.degrees(angulo):.1f}°")
        plt.xlabel("Distância horizontal (m)")
        plt.ylabel("Altura (m)")
        plt.title("Trajetória do projétil com arrasto")
        plt.grid(True)
        plt.legend()
        plt.show()
