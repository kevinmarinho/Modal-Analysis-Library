import pandas as pd
import scipy.linalg
import sympy as sp
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq, next_fast_len
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

def material_prop(material):
    if material == 'aluminio':
        E = 71.e9
        nu = 0.33
        r = 2.71e3
                
        L = 271.e-3
        a = 50.95e-3
        b = 3.16e-3
        Iy = a*b**3/12
        Iz = b*a**3/12
        A = area = a*b

    elif material ==  'aco':
        E = 200.e9
        nu = 0.3
        r = 7.84e3

        L = 285.e-3
        a = 51.25e-3
        b = 3.04e-3
        Iy = a*b**3/12
        Iz = b*a**3/12
        A = area = a*b

    elif material == 'acrilico':
        E = 20.e9
        nu = 0.25	
        r = 2e3
        
        L = 300.e-3
        a = 49e-3
        b = 3e-3
        Iy = a*b**3/12
        Iz = b*a**3/12
        A = a*b

    G = E/2/(1+nu)
    J = Iy + Iz

    return E, nu, r, L, a, b, Iy, Iz, A, G, J

def contorno(cc, ndofs, number_of_points):
    if cc == "engastado_engastado":
        boundary = (*np.arange(0,6), *np.arange(ndofs*number_of_points-6, ndofs*number_of_points))
        free = [i for i in range(ndofs*number_of_points) if i not in boundary]
        free_dof = len(free)
        ponto_resposta = free_dof//2 - 3
    elif cc == "engastado_livre":
        boundary = np.arange(0,6) #graus de liberdade travados
        free = [i for i in range(ndofs*number_of_points) if i not in boundary]
        free_dof = len(free)
        ponto_resposta = free_dof - 2
    elif cc == "livre":
        boundary = () #graus de liberdade travados
        free = [i for i in range(ndofs*number_of_points) if i not in boundary]
        free_dof = len(free)

    return boundary, free, free_dof, ponto_resposta
################# PARTE EXPERIMENTAL ###################

def leitura_experimental(material, cc):
    if cc == "engastado_engastado":
        if material == 'aluminio':
            nome_arquivo = 'exp_5/Al_biengastado_mcentro_acentro.csv'
        elif material ==  'aco':
            nome_arquivo = 'exp_5/Aco_biengastado_mcentro_acentro.csv'
        elif material == 'acrilico':
            nome_arquivo = 'exp_5/Acrilico_biengastado_mforaponta_aponta.csv'  
    elif cc == "engastado_livre":
        if material == 'aluminio':
            nome_arquivo = 'exp_5/Al_engastado_livre_mponta_aponta.csv'         
        elif material ==  'aco':
            nome_arquivo = 'exp_5/Aco_engastado_livre_mponta_aponta.csv'       
        elif material == 'acrilico':
            nome_arquivo = 'exp_5/Acrilico_engastado_livre_mponta_aponta.csv'
    with open(nome_arquivo, 'r') as arquivo:
        experimental = pd.read_csv(arquivo,
            sep=';', decimal=',', dtype=np.float64,
            skiprows=4, header=0, 
            names=['t1','r1','t2','r2','t3','r3'],
            parse_dates=['t1','t2','t3'],
            date_format="%f")
        
    exp_tempo = experimental['t1'].to_numpy(dtype=np.float64)
    exp_aceleracao = experimental['r1'].to_numpy(dtype=np.float64)
    exp_martelo = experimental['r2'].to_numpy(dtype=np.float64)

    return exp_tempo, exp_aceleracao, exp_martelo

def pos_experimental( material, exp_tempo, exp_aceleracao, exp_martelo, numero_de_pontos):
    N = numero_de_pontos

    time_increment = exp_tempo[1] - exp_tempo[0]
    DIFF = 0.05
    pos = 0

    for i in range(len(exp_aceleracao) - 1):
        if abs(exp_aceleracao[i + 1] - exp_aceleracao[i]) >= DIFF:
            pos = i
            break
    
    # pós-processamento dos gráficos

    tempo_cortado = np.array([i * time_increment for i in range(len(exp_tempo[pos:]))])
    aceleracao_cortada = exp_aceleracao[pos:]
    martelo_cortado = exp_martelo[pos:]

    
    amplitude = np.max(martelo_cortado)
    
    # Fast Fourier Transform

    T = tempo_cortado[1] - tempo_cortado[0]

    N = len(aceleracao_cortada)       
    transform_acel = fft(aceleracao_cortada)[:N//2]
    transform_martelo = fft(martelo_cortado)[:N//2]
    transform_freq = fftfreq(N, T)[:N//2]
    mask_freq = transform_freq < 500    
   
    
    freq_frf = transform_freq[mask_freq]
    ampl_frf = 2.0/N * np.abs(transform_acel[mask_freq]/transform_martelo[mask_freq])
    fase_frf = np.angle(transform_acel[mask_freq]/transform_martelo[mask_freq])*180/2/np.pi
     
    # encontrar os picos no gráfico da frequência  
    
    if material == 'aluminio':

        b, a = butter(N=7, Wn=1/2, btype='lowpass', analog=False, output='ba')
        signal_filtered = filtfilt(b, a, ampl_frf)

        peaks,__ = find_peaks(signal_filtered, distance=10, prominence =(np.max(ampl_frf) - np.min(ampl_frf))/1000)
    elif material == 'aco':
        b, a = butter(N=7, Wn=1/2, btype='lowpass', analog=False, output='ba')
        signal_filtered = filtfilt(b, a, ampl_frf)

        peaks,__ = find_peaks(signal_filtered, distance=50, prominence =(np.max(ampl_frf) - np.min(ampl_frf))/500)
    elif material == 'acrilico':
        b, a = butter(N=6, Wn=1/2, btype='lowpass', analog=False, output='ba')
        signal_filtered = filtfilt(b, a, ampl_frf)

        peaks,__ = find_peaks(signal_filtered, distance=50, prominence =(np.max(ampl_frf) - np.min(ampl_frf))/100)
   
    tempo_picos_freq = freq_frf[peaks]
    amp_picos_freq = ampl_frf[peaks]

    indices = np.argsort(amp_picos_freq)
    pico_1 = indices[-1]
    pico_2 = indices[-2]
    # encontrar os picos no gráfico da aceleração

    omega_1 = tempo_picos_freq[pico_1]
    omega_2 = tempo_picos_freq[pico_2]

    T_1 =  2*np.pi/(omega_1)
    T_2 = 2*np.pi/(omega_2)

    # primeiro modo

    distance_accel = (T_1*0.9)//time_increment    

    tempo_max = tempo_cortado[-1]
    peaks_2,__ = find_peaks(aceleracao_cortada, distance = distance_accel)

    tempo_picos = tempo_cortado[peaks_2]
    amp_picos = aceleracao_cortada[peaks_2]

    decrement_peaks = tempo_picos[tempo_picos>tempo_max/5][0:2]
    decrement_amp = amp_picos[tempo_picos>tempo_max/5][0:2]
    # segundo modo
    distance_accel_second_mode = T_2*0.9//time_increment

    peaks_3,__ = find_peaks(aceleracao_cortada, distance = distance_accel_second_mode)

    tempo_picos_second_mode = tempo_cortado[peaks_3]
    amp_picos_second_mode = aceleracao_cortada[peaks_3]

    decrement_peaks_second_mode = tempo_picos_second_mode[tempo_picos_second_mode>tempo_max/5][0:2]
    decrement_amp_second_mode = amp_picos_second_mode[tempo_picos_second_mode>tempo_max/5][0:2]
    ##
    

    # decremento logarítmico

    x_pico_1 = amp_picos[0]
    x_pico_2 = amp_picos[1]

    delta_1 = np.log(x_pico_1/x_pico_2)
    zeta_1 = delta_1/np.sqrt(4*np.pi**2 + delta_1**2)

    x_pico1_1 = amp_picos_second_mode [0]
    x_pico2_1 = amp_picos_second_mode [1]
    
    delta_2 = np.log(x_pico1_1/x_pico2_1)
    zeta_2 = delta_2/np.sqrt(4*np.pi**2 + delta_2**2)

    alfa = 2*omega_1*zeta_1
    
    beta = 2*zeta_2/omega_2

    phi_exp = freq_frf[peaks]

    numero_de_modos = len(phi_exp)

    # return exp_tempo[:pos], exp_aceleracao[:pos], exp_martelo[:pos], alfa, beta
    # return tempo_cortado, aceleracao_cortada, martelo_cortado, decrement_peaks, decrement_amp, freq_frf , ampl_frf , fase_frf , tempo_picos_freq, amp_picos_freq, amplitude
    return tempo_cortado, aceleracao_cortada, martelo_cortado, decrement_peaks, decrement_amp, decrement_peaks_second_mode, decrement_amp_second_mode, \
        freq_frf , ampl_frf, signal_filtered , fase_frf , tempo_picos_freq, amp_picos_freq, amplitude , alfa, beta, numero_de_modos , phi_exp

def pos_experimental_livre_livre(material, max_freq, max_time, numero_de_experimentos):

    for ind0 in range(numero_de_experimentos):
        if material == 'aco':
            if ind0 < 9:
                file = f'AÇO/LL_ACO_0{ind0 + 1}.csv'
            else:
                file = f'AÇO/LL_ACO_{ind0 + 1}.csv'
        if material == 'acrilico':
            if ind0 < 9:
                file = f'ACRILICO/LL_ACRILICO_0{ind0 + 1}.csv'
            else:
                file = f'ACRILICO/LL_ACRILICO_{ind0 + 1}.csv'
        if material == 'aluminio':
            file = f'Al/Al_livre_livre_{ind0 + 1}.csv'
        if material == 'fibra_vidro_0':
            file = f'Fibra vidro_ alinhado/livreLivre_fibravidro_{ind0 + 1}.csv'
        if material == 'fibra_vidro_90':
            if ind0 < 9:
                file = f'Fibra vidro_90/LL_FV_0{ind0 + 1}.csv'
            else:
                file = f'Fibra vidro_90/LL_FV_{ind0 + 1}.csv'
        elif material == 'fibra_carbono':
            file = f'FIBRA_CARBONO/FIB_CARBONO_LL_{ind0 + 1}.csv'  
        elif material == 'longarina_0':
            file = f'PLA_LONGARINA/PLA_LONGARINA_{ind0 + 1}.csv'
        elif material == 'longarina_90':
            file = f'PLA_LONGARINA_90/PLA_LONGARINA_90_{ind0 + 1}.csv'
        elif material == 'caverna':
            file = f'PLA_CAVERNA/PLA_CAVERNA_{ind0 + 1}.csv'  
        with open(file, 'r') as arquivo:
            experimental = pd.read_csv(arquivo,
                sep=';', decimal=',', dtype=np.float64,
                skiprows=4, header=0,
                names=['t1','a1','t2','a2','t3','h'],
                parse_dates=['t1','t2','t3'],
                date_format="%f")

        time = experimental['t1'].to_numpy(dtype=np.float64)
        accel_1 = experimental['a1'].to_numpy(dtype=np.float64) * 9.81
        accel_2 = experimental['a2'].to_numpy(dtype=np.float64) * 9.81
        hammer = experimental['h'].to_numpy(dtype=np.float64)

        time_increment = time[1] - time[0]
        diff_test = 0.05
        pos = 0

        for i in range(len(accel_1) - 1):
            if abs(accel_1[i + 1] - accel_1[i]) >= diff_test:
                pos = i
                break

        time = np.array([i * time_increment for i in range(len(time[pos:]))])
        if time[-1] > max_time:
            max_n = np.where(time < max_time)[0][-1]
        else:
            time = np.arange(0., max_time, time_increment)
            max_n = len(time)
        time = time[:max_n]
        accel_1 = accel_1[pos:]
        accel_2 = accel_2[pos:]
        hammer = hammer[pos:]
        accel_1 = accel_1[:max_n]
        accel_2 = accel_2[:max_n]
        hammer = hammer[:max_n]

        amplitude = np.max(hammer)

        T = time[1] - time[0]

        fs = 1./T
        cutoff = max_freq

        order = 4
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        accel_1 = filtfilt(b, a, accel_1)
        accel_2 = filtfilt(b, a, accel_2)
        hammer = filtfilt(b, a, hammer)

        N = len(accel_1)
        transform_accel_1 = fft(accel_1)[:N//2]
        transform_accel_2 = fft(accel_2)[:N//2]
        transform_hammer = fft(hammer)[:N//2]
        transform_freq = fftfreq(N, T)[:N//2]
        mask_freq = transform_freq < max_freq

        freq_frf = transform_freq[mask_freq]
        ampl_frf_1 = 2.0/N * np.abs(transform_accel_1[mask_freq]/transform_hammer[mask_freq])
        ampl_frf_2 = 2.0/N * np.abs(transform_accel_2[mask_freq]/transform_hammer[mask_freq])
        phas_frf_1 = np.angle(transform_accel_1[mask_freq]/transform_hammer[mask_freq])*180/2/np.pi
        phas_frf_2 = np.angle(transform_accel_2[mask_freq]/transform_hammer[mask_freq])*180/2/np.pi

        if ind0 == 0:
            ampl_frf_a1 = np.zeros(len(freq_frf))
            ampl_frf_a2 = np.zeros(len(freq_frf))
            phas_frf_a1 = np.zeros(len(freq_frf))
            phas_frf_a2 = np.zeros(len(freq_frf))

        ampl_frf_a1 += ampl_frf_1 / numero_de_experimentos
        ampl_frf_a2 += ampl_frf_2 / numero_de_experimentos
        phas_frf_a1 += phas_frf_1 / numero_de_experimentos
        phas_frf_a2 += phas_frf_2 / numero_de_experimentos

    frfs = [freq_frf, ampl_frf_a1, ampl_frf_a2, phas_frf_a1, phas_frf_a2]
    
    # peaks,__ = find_peaks(ampl_frf_a1, distance=10, prominence =(np.max(ampl_frf_a1) - np.min(ampl_frf_a1))/1000)
   
    # tempo_picos_freq = freq_frf[peaks]
    # amp_picos_freq = ampl_frf_a1[peaks]

    # indices = np.argsort(amp_picos_freq)
    # pico_1 = indices[-1]
    # pico_2 = indices[-2]
    # # encontrar os picos no gráfico da aceleração

    # omega_1 = tempo_picos_freq[pico_1]
    # omega_2 = tempo_picos_freq[pico_2]

    # T_1 =  2*np.pi/(omega_1)
    # T_2 = 2*np.pi/(omega_2)

    # # primeiro modo

    # distance_accel = (T_1*0.9)//time_increment    

    # tempo_max = time[-1]
    # peaks_2,__ = find_peaks(accel_1, distance = distance_accel)

    # tempo_picos = time[peaks_2]
    # amp_picos = accel_1[peaks_2]

    # decrement_peaks = tempo_picos[tempo_picos>tempo_max/5][0:2]
    # decrement_amp = amp_picos[tempo_picos>tempo_max/5][0:2]
    # # segundo modo
    # distance_accel_second_mode = T_2*0.9//time_increment

    # peaks_3,__ = find_peaks(accel_1, distance = distance_accel_second_mode)

    # tempo_picos_second_mode = time[peaks_3]
    # amp_picos_second_mode = accel_1[peaks_3]

    # decrement_peaks_second_mode = tempo_picos_second_mode[tempo_picos_second_mode>tempo_max/5][0:2]
    # decrement_amp_second_mode = amp_picos_second_mode[tempo_picos_second_mode>tempo_max/5][0:2]

    # decremento logarítmico

    # x_pico_1 = amp_picos[0]
    # x_pico_2 = amp_picos[1]

    # delta_1 = np.log(x_pico_1/x_pico_2)
    # zeta_1 = delta_1/np.sqrt(4*np.pi**2 + delta_1**2)

    # x_pico1_1 = amp_picos_second_mode [0]
    # x_pico2_1 = amp_picos_second_mode [1]
    
    # delta_2 = np.log(x_pico1_1/x_pico2_1)
    # zeta_2 = delta_2/np.sqrt(4*np.pi**2 + delta_2**2)

    # alfa = 2*omega_1*zeta_1
    
    # beta = 2*zeta_2/omega_2

    # phi_exp = freq_frf[peaks]

    # numero_de_modos = len(phi_exp)

    # return exp_tempo[:pos], exp_aceleracao[:pos], exp_martelo[:pos], alfa, beta
    # return tempo_cortado, aceleracao_cortada, martelo_cortado, decrement_peaks, decrement_amp, freq_frf , ampl_frf , fase_frf , tempo_picos_freq, amp_picos_freq, amplitude
    # return time, accel_1, phas_frf_a1, accel_2, phas_frf_a2, ampl_frf_a1, ampl_frf_a2, hammer, decrement_peaks, decrement_amp, decrement_peaks_second_mode, decrement_amp_second_mode, \
    #     freq_frf, tempo_picos_freq, amp_picos_freq, amplitude, alfa, beta, numero_de_modos, phi_exp
    return time, accel_1, phas_frf_a1, accel_2, phas_frf_a2, ampl_frf_a1, ampl_frf_a2, hammer, \
        freq_frf, amplitude

################# PARTE NUMÉRICA ###################

def matrixes():
    x = sp.symbols('x', real=True)
    E,A,Iy,Iz,G,L,r,J = sp.symbols('E,A,Iy,Iz,G,L,r,J', real=True, positive=True)

    phi = sp.Array([x/L, 1-x/L])
    dphi = sp.diff(phi, x)

    N = sp.Matrix([
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (1, L, L**2, L**3,),
        (0, 1, 2*L, 3*L**2,),
    ])

    psi = []
    for i in range(4):
        b = [0,0,0,0]
        b[i] = 1
        a = N.inv() * sp.Matrix(b)
        psi.append(a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3)

    psi = sp.Matrix(psi)

    dpsi = sp.diff(psi,x)
    ddpsi = sp.diff(psi,x,x)

    Bn = sp.Matrix([dphi[0], 0, 0, 0, 0, 0, dphi[1], 0, 0, 0, 0, 0,])
    Bv = sp.Matrix([0, ddpsi[0], ddpsi[1], 0, 0, 0, 0, ddpsi[2], ddpsi[3], 0, 0, 0])
    Bw = sp.Matrix([0, 0, 0, ddpsi[0], ddpsi[1], 0, 0, 0, 0, ddpsi[2], ddpsi[3], 0])
    Bt = sp.Matrix([0, 0, 0, 0, 0, dphi[0], 0, 0, 0, 0, 0, dphi[1]])

    Nu = sp.Matrix([phi[0], 0, 0, 0, 0, 0, phi[1], 0, 0, 0, 0, 0,])
    Nv = sp.Matrix([0, psi[0], psi[1], 0, 0, 0, 0, psi[2], psi[3], 0, 0, 0])
    Nw = sp.Matrix([0, 0, 0, psi[0], psi[1], 0, 0, 0, 0, psi[2], psi[3], 0])
    Nt = sp.Matrix([0, 0, 0, 0, 0, phi[0], 0, 0, 0, 0, 0, phi[1]])

    k = E*A*sp.integrate(Bn * Bn.T, (x, 0, L)) + \
        E*Iy*sp.integrate(Bw * Bw.T, (x, 0, L)) + \
        E*Iz*sp.integrate(Bv * Bv.T, (x, 0, L)) + \
        G*J*sp.integrate(Bt * Bt.T, (x, 0, L))

    m = r*A*sp.integrate(Nu * Nu.T, (x, 0, L)) + \
        r*A*sp.integrate(Nv * Nv.T, (x, 0, L)) + \
        r*A*sp.integrate(Nw * Nw.T, (x, 0, L)) + \
        r*Iz*sp.integrate(Nv * Nv.T, (x, 0, L)) + \
        r*Iy*sp.integrate(Nw * Nw.T, (x, 0, L)) + \
        r*J*sp.integrate(Nt * Nt.T, (x, 0, L))

    local_stiffness = sp.lambdify((E,A,Iy,Iz,G,L,r,J), k)
    local_mass = sp.lambdify((E,A,Iy,Iz,G,L,r,J), m)

    return local_stiffness, local_mass
    
def calculo_numerico(material, cc, alfa, beta, numero_de_modos, amplitude, numero_de_pontos, t_max):
    ndofs = 6
    local_stiffness, local_mass = matrixes()
    
    number_of_points = N = numero_de_pontos
    dt = 1e-4
    time = np.linspace(0,t_max,int(t_max/dt))

    E, _, r, L, _, _, Iy, Iz, A, G, J = material_prop(material)

    if numero_de_modos>5:
        numero_de_modos = 5
    _, free, free_dof, ponto_resposta = contorno(cc, ndofs, number_of_points)

    x = np.linspace(0., L, N)

    length = x[1] - x[0]
    k = local_stiffness(E,A,Iy,Iz,G,length,r,J)
    m = local_mass(E,A,Iy,Iz,G,length,r,J)

    stiffness = np.zeros((ndofs*N, ndofs*N))
    mass = np.zeros((ndofs*N, ndofs*N))
    for i in range(N-1):
        index = np.arange(ndofs*i, ndofs*i+12)
        stiffness[np.ix_(index, index)] += k
        mass[np.ix_(index, index)] += m

    damp = alfa * mass + beta * stiffness

    stiffness_free = stiffness[np.ix_(free, free)]
    mass_free = mass[np.ix_(free, free)]
    damp_free = damp[np.ix_(free, free)]

    mass_inv = np.linalg.inv(mass_free)

    stiff_block = - mass_inv @ stiffness_free
    damp_block = - mass_inv @ damp_free
    zeros_NN = np.zeros((free_dof, free_dof))
    eye_N = np.eye(free_dof)

    A = np.block( [[zeros_NN, eye_N], [stiff_block, damp_block]] )
    B = np.block([[zeros_NN], [mass_inv]])

    def force(t, amplitude):
        if hasattr(t, "__len__"):
            f = np.zeros(len(t))
            mask = t < 1e-3
            f[mask] = amplitude
            return f
        else:
            if t < 1e-3:
                return amplitude
            else:
                return 0

    def estado(t, y):
        f = np.zeros((len(A)//2))
        if cc == 'engastado_engastado':
            f[free_dof//2 - 3] = force(t, amplitude)

        elif cc=='engastado_livre':
            f[-3] = force(t, amplitude)

        dy = A @ y + B @ f

        return dy

    initial = np.zeros((len(A)))

    times = [0., t_max]

    sol = solve_ivp(estado, y0=initial, t_span=times, method='Radau', dense_output=True)

    results = sol.sol(time)
    v_num = results[free_dof + ponto_resposta]
    a_num = np.gradient(v_num, time)

    phi_num, _ = scipy.linalg.eigh(stiffness_free, b=mass_free, subset_by_index=[0, numero_de_modos-1])

    phi_num = np.sqrt(np.array(phi_num))
    
    return sol, a_num, ponto_resposta, phi_num

def calculo_numerico_modal(material, cc, zetas, numero_de_modos, amplitude, numero_de_pontos, t_max, dt):
    
    local_stiffness, local_mass = matrixes()
    
    number_of_points = N = numero_de_pontos
    time = np.linspace(0,t_max,int(t_max/dt))

    E, _, r, L, _, _, Iy, Iz, A, G, J = material_prop(material)
    ndofs = 6

    if numero_de_modos>5:
        numero_de_modos = 5

    _, free, free_dof, ponto_resposta = contorno(cc, ndofs, number_of_points)

    x = np.linspace(0., L, N)

    length = x[1] - x[0]
    k = local_stiffness(E,A,Iy,Iz,G,length,r,J)
    m = local_mass(E,A,Iy,Iz,G,length,r,J)

    stiffness = np.zeros((ndofs*N, ndofs*N))
    mass = np.zeros((ndofs*N, ndofs*N))
    for i in range(N-1):
        index = np.arange(ndofs*i, ndofs*i+12)
        stiffness[np.ix_(index, index)] += k
        mass[np.ix_(index, index)] += m

    stiffness_free = stiffness[np.ix_(free, free)]
    mass_free = mass[np.ix_(free, free)]
    mass_inv = np.linalg.inv(mass_free)

    damp_tilde = np.zeros((len(zetas),len(zetas)))
    for i,z in enumerate(zetas):
        damp_tilde[i,i] = z

    phi_num, phi = scipy.linalg.eigh(stiffness_free, b=mass_free, subset_by_index=[0, len(zetas)-1])
    damp_free = (phi @ damp_tilde) @ phi.T

    stiff_block = - mass_inv @ stiffness_free
    damp_block = - mass_inv @ damp_free
    zeros_NN = np.zeros((free_dof, free_dof))
    zeros_N = np.zeros((free_dof))
    eye_N = np.eye(free_dof)

    A = np.block( [[zeros_NN, eye_N], [stiff_block, damp_block]] )
    B = np.block([[zeros_NN], [mass_inv]])

    def force(t, amplitude):
        if hasattr(t, "__len__"):
            f = np.zeros(len(t))
            mask = t < 1e-3
            f[mask] = amplitude
            return f
        else:
            if t < 1e-3:
                return amplitude
            else:
                return 0

    def estado(t, y):
        f = np.zeros((len(A)//2))
        if cc == 'engastado_engastado':
            f[free_dof//2 - 3] = force(t, amplitude)

        elif cc=='engastado_livre':
            f[-3] = force(t, amplitude)

        dy = A @ y + B @ f

        return dy

    initial = np.zeros((len(A)))

    times = [0., t_max]

    sol = solve_ivp(estado, y0=initial, t_span=times, method='Radau', dense_output=True)

    results = sol.sol(time)
    v_num = results[free_dof + ponto_resposta]
    a_num = np.gradient(v_num, time)

    phi_num = np.sqrt(np.array(phi_num))
    
    return sol, a_num, ponto_resposta, phi_num

################# VETOR FORÇA ##################

def force(t, amplitude):
        if t<0.001:
            return amplitude
        else:
            return 0.

################# INTEGRAÇÃO ###################

def frf_tempo(exp_tempo, sol, ponto_resposta, amplitude):

    t_num = exp_tempo
    y_num = sol.sol(t_num)

    f_num = np.array([force(i,amplitude) for i in t_num])
    v_num = y_num[ponto_resposta]

    a_num = np.gradient(v_num, t_num)

    # Number of sample points
    N_num = len(a_num)
    # sample spacing
    T = exp_tempo[1] - exp_tempo[0]
    x = np.linspace(0.0, N_num*T, N_num, endpoint=False)
    yf = fft(a_num)[:N_num//2]
    yf2 = fft(f_num)[:N_num//2]
    xf = fftfreq(N_num, T)[:N_num//2]
    mask_freq = xf<500

    freq_frf_num = xf[mask_freq]
    ampl_frf_num = 2.0/N_num * np.abs(yf[mask_freq]/yf2[mask_freq])    
    fase_frf_num  = np.angle(yf[mask_freq]/yf2[mask_freq]*180/2/np.pi)

    return freq_frf_num, ampl_frf_num, fase_frf_num

def frf_direta(cc, material, amplitude, min_freq, max_freq, n_freq, t_max, dt, alfa, beta):
    number_of_points = N = 6
    
    local_stiffness, local_mass = matrixes()

    n_freq = 4000

    ndofs = 6

    _, free, free_dof, ponto_resposta = contorno(cc, ndofs, number_of_points)

    E, _, r, L, _, _, Iy, Iz, A, G, J = material_prop(material)
    
    x = np.linspace(0., L, N)

    length = x[1] - x[0]
    k = local_stiffness(E,A,Iy,Iz,G,length,r,J)
    m = local_mass(E,A,Iy,Iz,G,length,r,J)

    stiffness = np.zeros((ndofs*N, ndofs*N))
    mass = np.zeros((ndofs*N, ndofs*N))
    for i in range(N-1):
        index = np.arange(ndofs*i, ndofs*i+12)
        stiffness[np.ix_(index, index)] += k
        mass[np.ix_(index, index)] += m

    damp = alfa * mass + beta * stiffness

    stiffness_free = stiffness[np.ix_(free, free)]
    mass_free = mass[np.ix_(free, free)]
    damp_free = damp[np.ix_(free, free)]

    mass_inv = np.linalg.inv(mass_free)

    freq = np.linspace(min_freq, max_freq, n_freq)
    omega = freq/2/np.pi

    F = np.zeros(free_dof)
    if cc == 'engastado_engastado':
        F[free_dof//2 - 3] = amplitude

    elif cc=='engastado_livre':
        F[-3] = amplitude

    U = np.zeros(n_freq, dtype=np.complex128)
    u = np.zeros(n_freq, dtype=np.complex128)
    v = np.zeros(n_freq, dtype=np.complex128)
    a = np.zeros(n_freq, dtype=np.complex128)
    for i in range(len(omega)):
        w = omega[i]
        U = np.linalg.solve(- mass_free * w**2. + damp_free * w * 1j + stiffness_free, F)
        u[i] = U[ponto_resposta]
        v[i] = U[ponto_resposta] * w * 1j
        a[i] = U[ponto_resposta] * -w**2

    return freq,u,v,a

def frf_direta_modal(cc, material, amplitude, min_freq, max_freq, n_freq, t_max, dt, zetas, number_of_points):
    N = number_of_points

    local_stiffness, local_mass = matrixes()

    min_freq = 0
    max_freq = 500
    n_freq = 4000

    ndofs = 6

    _, free, free_dof, ponto_resposta = contorno(cc, ndofs, number_of_points)

    E, _, r, L, _, _, Iy, Iz, A, G, J = material_prop(material)
    
    x = np.linspace(0., L, N)

    length = x[1] - x[0]
    k = local_stiffness(E,A,Iy,Iz,G,L/(N-1),r,J)
    m = local_mass(E,A,Iy,Iz,G,L/(N-1),r,J)

    stiffness = np.zeros((ndofs*N, ndofs*N))
    mass = np.zeros((ndofs*N, ndofs*N))
    for i in range(N-1):
        index = np.arange(ndofs*i, ndofs*i+12)
        stiffness[np.ix_(index, index)] += k
        mass[np.ix_(index, index)] += m

    stiffness_free = stiffness[np.ix_(free, free)]
    mass_free = mass[np.ix_(free, free)]

    damp_tilde = np.zeros((len(zetas),len(zetas)))
    for i,z in enumerate(zetas):
        damp_tilde[i,i] = z

    _, phi = scipy.linalg.eigh(stiffness_free, b=mass_free, subset_by_index=[0, len(zetas)-1])
    damp_free = (phi @ damp_tilde) @ phi.T

    freq = np.linspace(min_freq, max_freq, n_freq)
    omega = 2*np.pi*freq

    F = np.zeros(free_dof)
    if cc == 'engastado_engastado':
        F[free_dof//2 - 3] = amplitude

    elif cc=='engastado_livre':
        F[-3] = amplitude

    U = np.zeros(n_freq, dtype=np.complex128)
    u = np.zeros(n_freq, dtype=np.complex128)
    v = np.zeros(n_freq, dtype=np.complex128)
    a = np.zeros(n_freq, dtype=np.complex128)
    for i in range(len(omega)):
        w = omega[i]
        U = np.linalg.solve(- mass_free * w**2. + damp_free * w * 1j + stiffness_free, F)
        u[i] = U[ponto_resposta]
        v[i] = U[ponto_resposta] * w * 1j
        a[i] = U[ponto_resposta] * -w**2

    return freq, u, v, a

def frf_direta_modal_comparacao(cc, material, amplitude,freq_experimental, zetas, number_of_points):
    N = number_of_points

    local_stiffness, local_mass = matrixes()

    min_freq = 0
    max_freq = 500
    n_freq = 4000

    ndofs = 6

    _, free, free_dof, ponto_resposta = contorno(cc, ndofs, number_of_points)

    E, _, r, L, _, _, Iy, Iz, A, G, J = material_prop(material)
    
    x = np.linspace(0., L, N)

    length = x[1] - x[0]
    k = local_stiffness(E,A,Iy,Iz,G,L/(N-1),r,J)
    m = local_mass(E,A,Iy,Iz,G,L/(N-1),r,J)

    stiffness = np.zeros((ndofs*N, ndofs*N))
    mass = np.zeros((ndofs*N, ndofs*N))
    for i in range(N-1):
        index = np.arange(ndofs*i, ndofs*i+12)
        stiffness[np.ix_(index, index)] += k
        mass[np.ix_(index, index)] += m

    stiffness_free = stiffness[np.ix_(free, free)]
    mass_free = mass[np.ix_(free, free)]

    damp_tilde = np.zeros((len(zetas),len(zetas)))
    for i,z in enumerate(zetas):
        damp_tilde[i,i] = z

    _, phi = scipy.linalg.eigh(stiffness_free, b=mass_free, subset_by_index=[0, len(zetas)-1])
    damp_free = (phi @ damp_tilde) @ phi.T

    freq = np.linspace(min_freq, max_freq, n_freq)
    freq = freq_experimental
    omega = 2*np.pi*freq

    F = np.zeros(free_dof)
    if cc == 'engastado_engastado':
        F[free_dof//2 - 3] = amplitude

    elif cc=='engastado_livre':
        F[-3] = amplitude

    U = np.zeros(n_freq, dtype=np.complex128)
    u = np.zeros(n_freq, dtype=np.complex128)
    v = np.zeros(n_freq, dtype=np.complex128)
    a = np.zeros(n_freq, dtype=np.complex128)
    for i in range(len(omega)):
        w = omega[i]
        U = np.linalg.solve(- mass_free * w**2. + damp_free * w * 1j + stiffness_free, F)
        u[i] = U[ponto_resposta]
        v[i] = U[ponto_resposta] * w * 1j
        a[i] = U[ponto_resposta] * -w**2

    return freq,u,v,a

def mac(phi_num, phi_exp):
    tamanho = np.min((len(phi_num), len(phi_exp)))
    phi_num = phi_num[:tamanho]
    phi_exp = phi_exp[:tamanho]

    MAC =  np.abs(np.dot(phi_num, np.conjugate(phi_exp))**2.) / \
        np.dot(phi_num, np.conjugate(phi_num)) / np.dot(phi_exp, np.conjugate(phi_exp))
    return MAC

def tsac(aceleracao_cortada, resultado_num):
    tamanho_exp = len(aceleracao_cortada)
    tamanho_num = len(resultado_num)

    menor_tamanho = np.min((tamanho_exp, tamanho_num))

    resultado_num = resultado_num[:menor_tamanho]
    aceleracao_cortada = aceleracao_cortada[:menor_tamanho]

    TSAC = np.dot(resultado_num, aceleracao_cortada)**2./np.dot(resultado_num, resultado_num)/np.dot(aceleracao_cortada, aceleracao_cortada)
    return TSAC
    
    
def rvac(ampl_frf, ampl_frf_num):

    RVAC = (np.abs(ampl_frf_num.T @ np.conj(ampl_frf)))**2./(ampl_frf_num.T @ np.conj(ampl_frf_num))/(ampl_frf.T @ np.conj(ampl_frf))

    return RVAC



def otimizador(metrica, material, cc, amplitude, numero_de_pontos, t_max):
    # if metrica=='tsac':
    #     sol = sc.minimize(tsac, args=(material, cc, amplitude, numero_de_pontos, t_max))
    # elif metrica=='mac':
    #     sol = sc.minimize(mac, )
    # return sol
    pass


# def plot_tempo(axis, x, y):
#     axis.plot(x, y)
#     pass

# def plot_freq():
#     pass