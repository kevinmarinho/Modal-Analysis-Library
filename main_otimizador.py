from Biblioteca import *
from scipy import optimize

material = 'acrilico'
cc = 'engastado_livre'

## EXPERIMENTAL

input_tempo, input_aceleracao, input_martelo= leitura_experimental(material, cc)

exp_tempo, exp_aceleracao, exp_martelo, decrement_peaks, decrement_amp, decrement_peaks_second_mode,\
    decrement_amp_second_mode, freq_frf, ampl_frf, fase_frf, tempo_picos_freq, amp_picos_freq, amplitude, alfa, beta, numero_de_modos, phi_exp = pos_experimental(material,  input_tempo, input_aceleracao, input_martelo, numero_de_pontos = 6)

exp_tempo_reduzido = exp_tempo[::4]

dt = exp_tempo[1] - exp_tempo[0]

zetas = [1e-6 for i in range(numero_de_modos)]

## NUMÉRICO
#
vetor_force = np.array( [force(i, amplitude) for i in exp_tempo])
sol, resultado, ponto_resposta , phi_num = \
    calculo_numerico_modal(material, cc, zetas, numero_de_modos, amplitude, 6, exp_tempo[-1], dt)


t_num = exp_tempo
y_num = sol.sol(t_num)

f_num = np.array([force(i,amplitude) for i in t_num])
v_num = y_num[ponto_resposta]

a_num = np.gradient(v_num, t_num)

## FRF

freq_num, u_num_frf, v_num_frf, a_num_frf =  frf_direta_modal(cc, material, amplitude, 1, 500, 2000, 2, 1e-4, zetas, 6)
amplitude_num = np.abs(a_num_frf)
fase_num = np.angle(a_num_frf)

numero_de_modos = 6

def metrica_MAC(x):
    zetas = x

    _, resultado, _, phi_num = calculo_numerico_modal(material, cc, zetas, numero_de_modos, amplitude, \
            6,  exp_tempo[-1], dt)

    MAC = mac(phi_num, phi_exp)
    return -MAC

def metrica_TSAC(x):
    zetas = x

    _, resultado, _, _ = calculo_numerico_modal(material, cc, zetas, numero_de_modos, amplitude, \
            6,  exp_tempo[-1], dt)
            

    TSAC = tsac(exp_aceleracao, resultado)
    return -TSAC

def metrica_RVAC(x):

    zetas = x

    _,_,_,a = frf_direta_modal_comparacao(cc, material, amplitude, freq_frf, zetas, 6)
    amplitude_num = np.abs(a)/9810**2

    RVAC = rvac(amplitude_num, ampl_frf)
    return -RVAC


# bounds = optimize.Bounds([0. for i in range(numero_de_modos)], \
#     [1. for i in range(numero_de_modos)])
# sol = optimize.minimize(metrica_TSAC, x0=[1e-3 for i in range(numero_de_modos)], method='L-BFGS-B', bounds=bounds)
# # sol = optimize.minimize(metrica_TSAC, x0=[1e-3, 1e-3], method='SLSQP', bounds=bounds)
# print(sol)

from geneticalgorithm2 import GeneticAlgorithm2 as ga
from geneticalgorithm2 import Generation, AlgorithmParams
from geneticalgorithm2 import get_population_initializer

var_bound = [(0., 1.e-5) for i in range(numero_de_modos)]
algorithm_parameters=AlgorithmParams(
    max_num_iteration=20,
    population_size=8,
    mutation_probability=0.1,
    mutation_discrete_probability=None,
    elit_ratio=0.01,
    parents_portion=0.3,
    crossover_type='uniform',
    mutation_type='uniform_by_center',
    mutation_discrete_type='uniform_discrete',
    selection_type='roulette',
    max_iteration_without_improv=50,
 )

model = ga( 
    dimension = numero_de_modos, 
    variable_type = 'real', 
    variable_boundaries = var_bound,
    algorithm_parameters = algorithm_parameters,
)

result = model.run(function=metrica_RVAC)

# best candidate
print(result.variable)

a=1