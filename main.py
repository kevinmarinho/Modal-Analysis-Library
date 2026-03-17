# Importação de bibliotecas necessárias
from Biblioteca import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Definição de parâmetros iniciais
material = 'aluminio'          # Material da viga
cc = 'engastado_livre'         # Condição de contorno
numero_de_pontos = 6           # Número de pontos na discretização
numero_de_modos = 6            # Número de modos a considerar


# Carregar e processar dados experimentais
input_tempo, input_aceleracao, input_martelo = leitura_experimental(material, cc)
exp_tempo, exp_aceleracao, exp_martelo, _, _, _, _, freq_frf, ampl_frf, _, _, _, _, amplitude, _, _, _, _ = pos_experimental(material, input_tempo, input_aceleracao, input_martelo, numero_de_pontos)

t_max = exp_tempo[-1]          # Tempo máximo da simulação
dt = exp_tempo[1] - exp_tempo[0]  # Passo de tempo
# Converter amplitude para unidades consistentes (m/s²)
amplitude *= 9.81

# Definir coeficientes de amortecimento (zetas) para cada modo
zetas = np.array([1e-6 for _ in range(numero_de_modos)])

# Calcular a resposta no domínio do tempo
sol, a_num, ponto_resposta, phi_num = calculo_numerico_modal(material, cc, zetas, numero_de_modos, amplitude, numero_de_pontos, t_max, dt)
t_num = np.linspace(0, t_max, len(a_num))  # Grid de tempo da simulação

# Interpolação para alinhar a_num ao grid de exp_tempo
interp_func = interp1d(t_num, a_num, kind='linear', fill_value="extrapolate")
a_num_interp = interp_func(exp_tempo)

# Plot da Aceleração no Tempo
plt.figure(figsize=(10, 6))
plt.plot(exp_tempo, exp_aceleracao, label='Experimental', color='tab:orange', linewidth=2)
plt.plot(exp_tempo, a_num_interp/9810, label='Numérico', color='tab:blue', linewidth=2)  # Multiplicar por 9.81 se necessário
plt.xlabel('Tempo [s]')
plt.ylabel('Aceleração [m/s²]')
plt.title('Aceleração vs Tempo')
plt.legend()
plt.grid(True)

# Calcular a FRF numérica com amortecimento
freq_num, _, _, a_num_frf = frf_direta_modal(cc, material, amplitude, 1, 500, 2000, t_max, dt, zetas, numero_de_pontos)
amplitude_num = np.abs(a_num_frf)  # Magnitude da FRF numérica

# Plot da FRF
plt.figure(figsize=(10, 6))
plt.semilogy(freq_frf, ampl_frf, label='Experimental', color='tab:orange', linewidth=2)
plt.semilogy(freq_num, amplitude_num / 9810**2, label='Numérico (com amortecimento)', color='tab:blue', linewidth=2)
plt.xlabel('Frequência [Hz]')
plt.ylabel('Amplitude [m/s²/N]')
plt.title('Função de Resposta em Frequência (FRF)')
plt.legend()
plt.grid(True)

# Destacar regiões de ressonância (opcional, para visualização)
plt.axvspan(0, 100, alpha=0.1, color='green', label='Região 1 (Modo 1)')
plt.axvspan(100, 250, alpha=0.1, color='yellow', label='Região 2 (Modo 2)')
plt.axvspan(250, 500, alpha=0.1, color='red', label='Região 3 (Modos 3+)')

plt.legend()
plt.show()