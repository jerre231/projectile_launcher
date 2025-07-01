import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def calcularCoeficienteAtrito(raio):
    rho = 1.225 # densidade do ar
    cd = 0.47   # para uma esfera
    area = np.pi * raio**2

    return (0.5 * rho * cd * area)


class SimuladorProjetil:
    def __init__(self, **kwargs):
        """
        Inicializa o simulador com os parâmetros físicos.
        """
        self.m = kwargs.get("massa")
        self.k = calcularCoeficienteAtrito(kwargs.get("raio"))
        self.g = kwargs.get("gravidade")
        self.tF = kwargs.get("tempo_de_forca")
        self.w = kwargs.get("rotacao")  # não usado ainda, mas incluído como atributo

        self.angulo = kwargs.get("angulo")
        self.forca = kwargs.get("forca")
        self.estado_inicial = [] # [x0, y0, vx0, vy0]

    def velocidadeInicial(self):
        """
        Primeiro método a ser chamado, recebe os parametros iniciais e define v0.
        """
        vx0 = (self.forca * self.tF * np.cos(np.radians(self.angulo))) / self.m
        vy0 = (self.forca * self.tF * np.sin(np.radians(self.angulo))) / self.m

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
        self.velocidadeInicial()
        estado = self.estado_inicial
        xs = [estado[0]] # x0
        ys = [estado[1]] # y0
        iteracoes = 0
        min_iteracoes = 5
        tempo_total = 0

        v0 = np.sqrt(estado[2]**2 + estado[3]**2)

        while estado[1] >= 0 & iteracoes <= min_iteracoes: # enquanto y não for negativo, ou seja, para quando tocar o chão novamente
            estado = self.runge_kutta4(estado, dt)
            xs.append(estado[0])
            ys.append(estado[1])
            tempo_total += dt
            iteracoes += 1

        max_x = max(xs)
        max_y = max(ys)

        print(v0)
        print(self.angulo)
        print(tempo_total)

        plt.figure(figsize=(10, 6))
        plt.plot(xs, ys, label=f"v0 = {v0:.2f} m/s, ângulo = {self.angulo:.1f}°\ntempo: {tempo_total:.3f}s\nx max: {max_x:.3f}m\ny max: {max_y:.3f}m")
        plt.xlabel("Distância horizontal (m)")
        plt.ylabel("Altura (m)")
        plt.title("Trajetória do projétil com arrasto")

        plt.xlim(0, max_x * 1.1)
        plt.ylim(0, max_y * 1.1)
        plt.grid(True)
        plt.legend()
        plt.show()

    def plotar_trajetoria_animada(self, dt=0.01):
        """
        Simula a trajetória do projétil e cria uma animação.
        """
        # 1. Definir o estado inicial
        self.velocidadeInicial() # Chama para calcular self.estado_inicial
        estado = self.estado_inicial.copy() # Usa uma cópia para não modificar o original

        xs = [estado[0]] # x0
        ys = [estado[1]] # y0
        tempos = [0.0]   # tempo inicial
        
        # 2. Pré-calcular toda a trajetória
        max_iteracoes = 10000 # Limite para evitar loops infinitos em caso de erro
        iter_count = 0

        # Condição de parada: enquanto y for não negativo E não excedeu o limite de iterações
        while estado[1] >= 0 and iter_count < max_iteracoes:
            estado = self.runge_kutta4(estado, dt)
            xs.append(estado[0])
            ys.append(estado[1])
            tempos.append(tempos[-1] + dt)
            iter_count += 1
        
        # Ajuste final se o projétil passou do chão
        if ys[-1] < 0:
            # Interpolar para encontrar o ponto exato de impacto no chão
            # Este é um refinamento, mas para o plot simples, pode-se simplesmente cortar
            # Simplificação: Apenas remove o último ponto se ele for negativo
            if len(xs) > 1:
                xs.pop()
                ys.pop()
                tempos.pop()
            # Adiciona o ponto de impacto no chão se necessário (linear interpolação simples)
            # x_impacto = xs[-1] + (xs[-2] - xs[-1]) * (0 - ys[-1]) / (ys[-2] - ys[-1]) if len(xs) > 1 else xs[-1]
            # xs[-1] = x_impacto
            # ys[-1] = 0.0

        # Calcular informações da trajetória
        max_x = np.max(xs)
        max_y = np.max(ys) if ys else 0
        # Garantir que max_y não seja negativo e tenha um mínimo para o ylim
        if max_y < 0: max_y = 0
        if len(ys) > 1: max_y = max(ys)
        
        # A velocidade inicial é a calculada por velocidadeInicial()
        v0_inicial = np.sqrt(self.estado_inicial[2]**2 + self.estado_inicial[3]**2)
        tempo_total_voo = tempos[-1] if tempos else 0.0

        # --- Configurar a figura e o eixo para a animação ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, max_x * 1.1)
        ax.set_ylim(0, max_y * 1.2 if max_y > 0 else 10) # Garante um mínimo de 10m de altura se não houver altura

        ax.set_xlabel("Distância horizontal (m)")
        ax.set_ylabel("Altura (m)")
        ax.set_title("Animação da Trajetória do Projétil")
        ax.grid(True)

        # O "trail" mostra o caminho percorrido pelo projétil
        # Inicialmente vazio, será preenchido na animação
        line, = ax.plot([], [], 'o-', lw=2, color='blue', label='Trajetória')
        # O "point" é a posição atual do projétil
        point, = ax.plot([], [], 'o', markersize=8, color='red', label='Posição Atual')
        # Texto para exibir informações
        info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, verticalalignment='top') # Posição relativa

        ax.legend()

        # --- Função de Inicialização para o FuncAnimation ---
        def init_animation():
            line.set_data([], [])
            point.set_data([], [])
            info_text.set_text('')
            return line, point, info_text

        def animate_frame(i):
            # CORREÇÃO AQUI: Envolver xs[i] e ys[i] em listas para 'point'
            line.set_data(xs[:i+1], ys[:i+1])
            point.set_data([xs[i]], [ys[i]]) # AGORA ESTÁ CORRETO
            
            current_x = xs[i]
            current_y = ys[i]
            current_time = tempos[i]
            
            info_text.set_text(
                f'v0 = {v0_inicial:.2f} m/s\n'
                f'Ângulo = {self.angulo:.1f}°\n'
                f'Tempo de voo: {tempo_total_voo:.3f} s\n'
                f'Alcance máximo: {max_x:.3f} m\n'
                f'Altura máxima: {max_y:.3f} m\n'
                f'Tempo atual: {current_time:.2f} s\n'
                f'Posição: ({current_x:.2f}m, {current_y:.2f}m)'
            )
            return line, point, info_text

        ani = animation.FuncAnimation(
            fig, animate_frame, frames=len(xs), init_func=init_animation, 
            interval=dt*1000, blit=True, repeat=False
        )

        plt.show()

    def comparar_dt(self, dts):
        """
        Compara a trajetória do projétil para diferentes valores de dt.
        
        Parâmetros:
        - dts: lista de valores de passo de tempo (dt) a serem testados
        """
        plt.figure(figsize=(10, 6))
        iteracoes = 0
        min_iteracoes = 5
        todos_xs = []
        todos_ys = []

        for dt in dts:
            estado = self.estado_inicial.copy()
            tempo_total = 0
            xs = [estado[0]]
            ys = [estado[1]]
            v0 = np.sqrt(estado[2]**2 + estado[3]**2)

            while estado[1] >= 0 & iteracoes <= min_iteracoes:
                
                estado = self.runge_kutta4(estado, dt)
                xs.append(estado[0])
                ys.append(estado[1])
                todos_xs.append(xs)
                todos_ys.append(ys)
                tempo_total += dt
                iteracoes += 1

            print(tempo_total)
            plt.plot(xs, ys, label=f"dt = {dt:.2f}s\ntempo:{tempo_total}s")

        plt.xlabel("Distância horizontal (m)")
        plt.ylabel("Altura (m)")
        plt.title(f"Comparação de trajetórias para diferentes dt (ângulo = {self.angulo:.1f}°)")
        max_x = max([max(xs) for xs in todos_xs])
        max_y = max([max(ys) for ys in todos_ys])
        plt.xlim(0, max_x * 1.1)
        plt.ylim(0, max_y * 1.1)
        plt.grid(True)
        plt.legend(loc="best")
        plt.show()

    def comparar_configuracoes(self, configuracoes):
        """
        Compara diferentes trajetórias com base em múltiplas configurações de entrada.

        Parâmetro:
        - configuracoes: lista de dicionários, cada um contendo:
            {
                "forca": valor_em_newtons,
                "angulo": valor_em_graus,
                "dt": passo_de_tempo
            }
        """
        plt.figure(figsize=(10, 6))
        
        for config in configuracoes:
            self.velocidadeInicial(config["forca"], config["angulo"])
            estado = self.estado_inicial.copy()
            xs = [estado[0]]
            ys = [estado[1]]
            tempo = 0

            while estado[1] >= 0:
                estado = self.runge_kutta4(estado, config["dt"])
                xs.append(estado[0])
                ys.append(estado[1])
                tempo += config["dt"]

            v0 = np.sqrt(self.estado_inicial[2]**2 + self.estado_inicial[3]**2)
            plt.plot(xs, ys, label=f"forca= {config['forca']}N, angulo= {config['angulo']}°, dt={config['dt']}, v0={v0:.1f}m/s")

        plt.xlabel("Distância horizontal (m)")
        plt.ylabel("Altura (m)")
        plt.title("Comparação de trajetórias com diferentes configurações")
        plt.grid(True)
        plt.legend()
        plt.show()

