import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, CubicSpline
import time

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
        self.dt = kwargs.get("dt")

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
    
    def runge_kutta4_generico(self, f, estado, dt):
        """
        Executa um passo de Runge-Kutta 4ª ordem para uma função f(estado).
        """
        k1 = f(estado)
        k2 = f(estado + 0.5 * dt * k1)
        k3 = f(estado + 0.5 * dt * k2)
        k4 = f(estado + dt * k3)

        return estado + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


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

    def plotar_trajetoria(self):
        """
        Simula e plota a trajetória do projétil.
        """
        self.velocidadeInicial()
        dt = self.dt
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
    
    def rk45(self, t_max=5, num_pontos=3000):
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, num_pontos)

        def sistema(t, estado):
            return self.EDOs(estado)
        
        dt_max = 1/self.velocidadeInicial
        sol = solve_ivp(sistema, t_span, self.estado_inicial, t_eval=t_eval, method='RK45', max_step=dt_max)

        x = sol.y[0]
        y = sol.y[1]

        return x, y
    
    def plotar_trajetoria_animada_comparada(self, t_max=10, num_pontos_scipy=1000):
        """
        Anima a trajetória do projétil usando RK4 manual e plota a trajetória obtida com solve_ivp (RK45) ao final.
        Exibe também os erros ao final da animação.
        """

        # --- Parte 1: Simulação com Runge-Kutta 4 manual ---
        self.velocidadeInicial()
        dt = self.dt
        estado = self.estado_inicial.copy()

        xs = [estado[0]]
        ys = [estado[1]]
        tempos = [0.0]

        max_iteracoes = 10000
        iter_count = 0

        while estado[1] >= 0 and iter_count < max_iteracoes:
            estado = self.runge_kutta4(estado, dt)
            xs.append(estado[0])
            ys.append(estado[1])
            tempos.append(tempos[-1] + dt)
            iter_count += 1

        if ys[-1] < 0:
            xs.pop()
            ys.pop()
            tempos.pop()

        xs = np.array(xs)
        ys = np.array(ys)
        tempos = np.array(tempos)

        max_x = np.max(xs)
        max_y = np.max(ys) if ys.size > 0 else 0
        if max_y < 0: max_y = 0

        v0_inicial = np.sqrt(self.estado_inicial[2]**2 + self.estado_inicial[3]**2)
        tempo_total_voo = tempos[-1] if tempos.size > 0 else 0.0

        # --- Parte 2: Solução com solve_ivp (RK45) ---
        def sistema(t, estado):
            return self.EDOs(estado)

        t_eval = np.linspace(0, t_max, num_pontos_scipy)
        sol = solve_ivp(sistema, [0, t_max], self.estado_inicial.copy(), t_eval=t_eval, method='RK45')

        xs_scipy = sol.y[0]
        ys_scipy = sol.y[1]
        tempos_scipy = sol.t

        idx_queda = np.where(ys_scipy < 0)[0]
        if len(idx_queda) > 0:
            idx_fim = idx_queda[0]
            xs_scipy = xs_scipy[:idx_fim]
            ys_scipy = ys_scipy[:idx_fim]
            tempos_scipy = tempos_scipy[:idx_fim]

        # --- Parte 3: Calcular os erros ---
        interp_x = interp1d(tempos_scipy, xs_scipy, kind='linear', fill_value="extrapolate")
        interp_y = interp1d(tempos_scipy, ys_scipy, kind='linear', fill_value="extrapolate")

        xs_ref = interp_x(tempos)
        ys_ref = interp_y(tempos)

        erro_x = np.abs(xs - xs_ref)
        erro_y = np.abs(ys - ys_ref)
        erro_total = np.sqrt(erro_x**2 + erro_y**2)

        erro_max = np.max(erro_total)
        erro_medio = np.mean(erro_total)
        erro_rms = np.sqrt(np.mean(erro_total**2))

        # --- Parte 4: Animação ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, max(max_x, np.max(xs_scipy)) * 1.1)
        ax.set_ylim(0, max(max_y, np.max(ys_scipy)) * 1.2)
        ax.set_xlabel("Distância horizontal (m)")
        ax.set_ylabel("Altura (m)")
        ax.set_title("Animação da Trajetória do Projétil")
        ax.grid(True)

        line, = ax.plot([], [], 'o-', lw=2, color='blue', label='Trajetória (RK4 manual)')
        point, = ax.plot([], [], 'o', markersize=8, color='red', label='Posição Atual')
        rk45_line, = ax.plot([], [], '--', color='green', lw=2, label='Solução RK45 (solve_ivp)')
        spline_line, = ax.plot([], [], '-', color='darkred', lw=2, label='Spline RK4')
        info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, verticalalignment='top', fontsize=10)
        ax.legend()

        # --- Gerar spline da RK4 ---
        spline_x_rk4, spline_y_rk4 = self.gerar_spline(tempos, xs, ys)
        t_suave = np.linspace(tempos[0], tempos[-1], 1000)
        xs_suave = spline_x_rk4(t_suave)
        ys_suave = spline_y_rk4(t_suave)

        def init_animation():
            line.set_data([], [])
            point.set_data([], [])
            rk45_line.set_data([], [])
            spline_line.set_data([], [])
            info_text.set_text('')
            return line, point, info_text, rk45_line, spline_line

        def animate_frame(i):
            line.set_data(xs[:i+1], ys[:i+1])
            point.set_data([xs[i]], [ys[i]])
            current_x = xs[i]
            current_y = ys[i]
            current_time = tempos[i]

            texto = (
                f'v0 = {v0_inicial:.2f} m/s\n'
                f'Ângulo = {self.angulo:.1f}°\n'
                f'Tempo de voo (RK4): {tempo_total_voo:.3f} s\n'
                f'Alcance máximo (RK4): {max_x:.3f} m\n'
                f'Altura máxima (RK4): {max_y:.3f} m\n'
                f'Tempo atual: {current_time:.2f} s\n'
                f'Posição: ({current_x:.2f}m, {current_y:.2f}m)'
            )

            if i == len(xs) - 1:
                rk45_line.set_data(xs_scipy, ys_scipy)
                spline_line.set_data(xs_suave, ys_suave)
                texto += (
                    f'\n\nErro máximo: {erro_max:.4e} m\n'
                    f'Erro médio: {erro_medio:.4e} m\n'
                    f'Erro RMS: {erro_rms:.4e} m'
                )

            info_text.set_text(texto)
            return line, point, info_text, rk45_line, spline_line

        ani = animation.FuncAnimation(
            fig, animate_frame, frames=len(xs), init_func=init_animation,
            interval=dt * 1000, blit=True, repeat=False
        )

        plt.show()

    def calcular_erros(xs_rk4, ys_rk4, tempos_rk4, xs_rk45, ys_rk45, tempos_rk45):
        """
        Calcula erros entre a solução RK4 manual e a solução de referência RK45 (solve_ivp).
        """
        # Spline cúbica da sua RK4 manual
        spline_x_rk4 = CubicSpline(tempos_rk4, xs_rk4)
        spline_y_rk4 = CubicSpline(tempos_rk4, ys_rk4)

        # Interpola a RK4 nos tempos da referência
        xs_interp = spline_x_rk4(tempos_rk45)
        ys_interp = spline_y_rk4(tempos_rk45)

        # Erros ponto a ponto
        erro_x = np.abs(xs_interp - xs_rk45)
        erro_y = np.abs(ys_interp - ys_rk45)
        erro_total = np.sqrt(erro_x**2 + erro_y**2)

        # Métricas
        erro_max = np.max(erro_total)
        erro_medio = np.mean(erro_total)
        erro_rms = np.sqrt(np.mean(erro_total**2))

        return {
            'erro_max': erro_max,
            'erro_medio': erro_medio,
            'erro_rms': erro_rms,
            'erro_total': erro_total,  # útil se quiser plotar ao longo do tempo
            'tempos': tempos_rk45
        }
    def gerar_spline(self, tempos, xs, ys):
        """
        Gera splines cúbicas para interpolar a trajetória do projétil.
        Retorna duas funções spline_x(t) e spline_y(t).
        """
        spline_x = CubicSpline(tempos, xs)
        spline_y = CubicSpline(tempos, ys)
        return spline_x, spline_y
    def comparar_metodos(self, t_max=10, num_pontos_scipy=1000, latitude_graus=-22.9):
        """
        Compara a trajetória do projétil resolvida com os métodos:
        - Euler
        - Runge-Kutta 4 (manual)
        - RK45 (solve_ivp - referência)
        
        Plota as trajetórias e exibe os erros dos métodos numéricos em relação à solução de referência dentro do gráfico,
        além de calcular o desvio lateral causado pela força de Coriolis.
        """
        self.velocidadeInicial()
        dt = self.dt
        estado_inicial = self.estado_inicial.copy()

        # --- Euler ---
        estado_euler = estado_inicial.copy()
        xs_euler, ys_euler, tempos_euler = [estado_euler[0]], [estado_euler[1]], [0.0]

        timestart_euler = time.time()
        while estado_euler[1] >= 0 and tempos_euler[-1] <= t_max:
            derivadas = self.EDOs(estado_euler)
            estado_euler = estado_euler + dt * np.array(derivadas)
            xs_euler.append(estado_euler[0])
            ys_euler.append(estado_euler[1])
            tempos_euler.append(tempos_euler[-1] + dt)
        timeend_euler = time.time()
        time_euler = timeend_euler - timestart_euler

        # --- RK4 manual ---
        estado_rk4 = estado_inicial.copy()
        xs_rk4, ys_rk4, tempos_rk4 = [estado_rk4[0]], [estado_rk4[1]], [0.0]

        timestart_rk4 = time.time()
        while estado_rk4[1] >= 0 and tempos_rk4[-1] <= t_max:
            estado_rk4 = self.runge_kutta4(estado_rk4, dt)
            xs_rk4.append(estado_rk4[0])
            ys_rk4.append(estado_rk4[1])
            tempos_rk4.append(tempos_rk4[-1] + dt)
        timeend_rk4 = time.time()
        time_rk4 = timeend_rk4 - timestart_rk4

        # --- Solução de referência (RK45 com solve_ivp) ---
        def sistema(t, estado):
            return self.EDOs(estado)

        v0 = np.sqrt(self.estado_inicial[2]**2 + self.estado_inicial[3]**2)
        dt_max = 1/v0

        t_eval = np.linspace(0, t_max, num_pontos_scipy)

        timestart_rk45 = time.time()
        sol = solve_ivp(sistema, [0, t_max], estado_inicial.copy(), t_eval=t_eval, method='RK45', max_step=dt_max)
        timeend_rk45 = time.time()
        time_rk45 = timeend_rk45 - timestart_rk45

        xs_ref, ys_ref, tempos_ref = sol.y[0], sol.y[1], sol.t

        idx_queda = np.where(ys_ref < 0)[0]
        if len(idx_queda) > 0:
            idx_fim = idx_queda[0]
            xs_ref = xs_ref[:idx_fim]
            ys_ref = ys_ref[:idx_fim]
            tempos_ref = tempos_ref[:idx_fim]

        # --- Interpolação para comparação de erros ---
        interp_x_ref = interp1d(tempos_ref, xs_ref, kind='linear', fill_value="extrapolate")
        interp_y_ref = interp1d(tempos_ref, ys_ref, kind='linear', fill_value="extrapolate")

        def calcular_erros(tempos, xs, ys):
            xs_interp = interp_x_ref(tempos)
            ys_interp = interp_y_ref(tempos)
            erro = np.sqrt((xs - xs_interp)**2 + (ys - ys_interp)**2)
            return {
                'erro_max': np.max(erro),
                'erro_medio': np.mean(erro),
                'erro_rms': np.sqrt(np.mean(erro**2))
            }

        erros_euler = calcular_erros(np.array(tempos_euler), np.array(xs_euler), np.array(ys_euler))
        erros_rk4   = calcular_erros(np.array(tempos_rk4), np.array(xs_rk4), np.array(ys_rk4))

        # --- Cálculo do desvio de Coriolis ---
        desvio_coriolis = self.calcular_desvio_final_coriolis(latitude_graus)

        # --- Plot ---
        plt.figure(figsize=(11, 6))
        plt.plot(xs_ref, ys_ref, '--', label='RK45 (solve_ivp)', color='green')
        plt.plot(xs_rk4, ys_rk4, '-', label='RK4 (manual)', color='blue')
        plt.plot(xs_euler, ys_euler, '-', label='Euler', color='orange')
        plt.xlabel("Distância horizontal (m)")
        plt.ylabel("Altura (m)")
        plt.title("Comparação dos Métodos Numéricos para a Trajetória do Projétil")
        plt.legend()
        plt.grid(True)

        # --- Texto de erro e desvio de Coriolis no gráfico ---
        texto_info = (
            "Erros em relação ao RK45 (solve_ivp):\n\n"
            "Método de Euler:\n"
            f"  Erro máximo: {erros_euler['erro_max']:.4e} m\n"
            f"  Erro médio:  {erros_euler['erro_medio']:.4e} m\n"
            f"  Erro RMS:    {erros_euler['erro_rms']:.4e} m\n\n"
            f"  Runtime:    {time_euler:.4f} s\n\n"
            "Método de RK4 manual:\n"
            f"  Erro máximo: {erros_rk4['erro_max']:.4e} m\n"
            f"  Erro médio:  {erros_rk4['erro_medio']:.4e} m\n"
            f"  Erro RMS:    {erros_rk4['erro_rms']:.4e} m\n\n"
            f"  Runtime:    {time_rk4:.4f} s\n\n"
            f"Desvio lateral por Coriolis (latitude {latitude_graus}°):\n"
            f"  z_final = {desvio_coriolis:.3f} m"
        )

        plt.text(1.02, 0.5, texto_info, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='center', family='monospace')

        plt.tight_layout()
        plt.show()


    def EDOs_com_coriolis(self, estado, latitude_graus):
        """
        Retorna as derivadas no sistema com efeito de Coriolis.
        Estado: [x, y, z, vx, vy, vz]
        """
        x, y, z, vx, vy, vz = estado
        v = np.array([vx, vy, vz])
        
        # Constantes
        g = self.g
        omega = self.w  # rad/s
        phi = np.radians(latitude_graus)
        
        # Vetor de rotação da Terra (depende da latitude)
        # No referencial local: x (leste), y (vertical), z (norte)
        Omega = omega * np.array([0, np.cos(phi), np.sin(phi)])  # vetor de rotação da Terra

        # Força de Coriolis
        a_coriolis = -2 * np.cross(Omega, v)

        # Força de arrasto não linear (se quiser incluir)
        norm_v = np.linalg.norm(v)
        a_drag = - (self.k * norm_v * v) / self.m

        # Aceleração total
        a_total = np.array([0, -g, 0]) + a_coriolis + a_drag

        return np.concatenate([v, a_total])
    def calcular_desvio_final_coriolis(self, latitude_graus=0):
        """
        Usa Runge-Kutta 4 genérico para integrar com Coriolis em 3D.
        Retorna o desvio lateral (z) ao final da trajetória.
        """
        self.velocidadeInicial()
        
        v0x = self.estado_inicial[2]
        v0y = self.estado_inicial[3]
        estado = np.array([0.0, 0.0, 0.0, v0x, v0y, 0.0])  # [x, y, z, vx, vy, vz]
        dt = self.dt

        posicoes = [estado[:3]]
        max_iter = 10000
        iter_count = 0

        f = lambda s: self.EDOs_com_coriolis(s, latitude_graus)

        while estado[1] >= 0 and iter_count < max_iter:
            estado = self.runge_kutta4_generico(f, estado, dt)
            posicoes.append(estado[:3])
            iter_count += 1

        posicoes = np.array(posicoes)
        return posicoes[-1, 2]  # z final
    
