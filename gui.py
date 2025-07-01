
"""
Objetivo -> Coletar os parâmetros para cada simulação
Parâmetros principais a serem coletados -> {angulo, força, dt}
Parâmetros com default mas podem ser mudados -> {massa, raio, gravidade, rotacao, tempo_de_forca}
"""

import tkinter as tk
from tkinter import *

# definindo o dicionário a ser utilizado para guardar os parâmetros
params = {
    'massa': None,
    'coeficiente_atrito': None,
    'gravidade': None,
    'rotacao': None,
    'tempo_de_forca': None,
    'raio': None,
    'angulo': None,
    'forca': None,
    'dt': None
}
def parameter_collect():

    # Criando a janela
    window = tk.Tk()
    window.title("Projectile Launcher - Set Parameters")
    frame = tk.Frame(window)
    frame.pack()

    # Entrada dos Parâmetros
    param_entry_frame = tk.LabelFrame(frame, text="Seleção de Parâmetros")
    param_entry_frame.grid(row= 0, column= 0, padx= 20, pady= 10)

    #Essenciais
    angulo_label = tk.Label(param_entry_frame, text="Ângulo de Lançamento (Graus)")
    angulo_label.grid(row=2, column=0)
    angulo_entry = tk.Entry(param_entry_frame)
    angulo_entry.grid(row=2, column=1)

    forca_label = tk.Label(param_entry_frame, text="Força de Lançamento (Newtons)")
    forca_label.grid(row=3, column=0)
    forca_entry = tk.Entry(param_entry_frame)
    forca_entry.grid(row=3, column=1)

    dt_label = tk.Label(param_entry_frame, text="Passo de Integração (segundos)")
    dt_label.grid(row=4, column=0)
    dt_entry = tk.Entry(param_entry_frame)
    dt_entry.grid(row=4, column=1)

    #Específicos
    massa_label = tk.Label(param_entry_frame, text="Massa do Projétil (Quilos)")
    massa_label.grid(row=1, column=2)
    massa_entry = tk.Entry(param_entry_frame)
    massa_entry.grid(row=1, column=3)
    massa_entry.insert(END, '1')

    raio_label =tk.Label(param_entry_frame, text="Raio do Projétil (Metros)")
    raio_label.grid(row=2, column=2)
    raio_entry =tk.Entry(param_entry_frame)
    raio_entry.grid(row=2, column=3)
    raio_entry.insert(END, '0.5')

    gravidade_label = tk.Label(param_entry_frame, text="Gravidade do Planeta (m/s^2)")
    gravidade_label.grid(row=3, column=2)
    gravidade_entry = tk.Entry(param_entry_frame)
    gravidade_entry.grid(row=3, column=3)
    gravidade_entry.insert(END, '9.807')

    rotacao_label = tk.Label(param_entry_frame, text="Rotação do Planeta (kg/m^3)")
    rotacao_label.grid(row=4, column=2)
    rotacao_entry = tk.Entry(param_entry_frame)
    rotacao_entry.grid(row=4, column=3)
    rotacao_entry.insert(END, '0.000073')

    tempoForca_label = tk.Label(param_entry_frame, text="Tempo de Apl. da Força (segundos)")
    tempoForca_label.grid(row=5, column=2)
    tempoForca_entry = tk.Entry(param_entry_frame)
    tempoForca_entry.grid(row=5, column=3)
    tempoForca_entry.insert(END, '0.0015')

    for widget in param_entry_frame.winfo_children():
        widget.grid_configure(padx=10, pady=5)

    # Função de coleta de dados
    def get_data():
        global data_submitted
        try:
            params["angulo"] = float(angulo_entry.get())
            params["forca"] = float(forca_entry.get())
            params["dt"] = float(dt_entry.get())

            params["massa"] = float(massa_entry.get())
            params["raio"] = float(raio_entry.get())
            params["gravidade"] = float(gravidade_entry.get())
            params["rotacao"] = float(rotacao_entry.get())
            params["tempo_de_forca"] = float(tempoForca_entry.get())
        
            print("Parâmetros coletados com sucesso:")
            for key, value in params.items():
                print(f"- {key}: {value}")

            data_submitted = True
            window.destroy()
        except ValueError:
            tk.messagebox.showerror("Erro de Entrada", "Por favor, insira valores numéricos válidos para todos os parâmetros.")
        except Exception as e:
            tk.messagebox.showerror("Erro", f"Ocorreu um erro: {e}")
    # Submit
    button = tk.Button(frame, text="Começar Simulação", command= get_data)
    button.grid(row=3, column=0, sticky="news", padx=20, pady=10)

    window.mainloop()
    return data_submitted

if __name__ == "__main__":
    if parameter_collect():
        print("\nParâmetros:")
        print(params)
    else:
        print("\nGUI fechada sem recolher dados.")