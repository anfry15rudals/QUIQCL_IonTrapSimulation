import multiprocessing
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import time

# User-defined parameters
n_tot = 5               # total number of number state basis to represent the motional state: starts from zero
n_mean_init = 0         # initial mean number of the motional state
tls_zero_init = 1      # initial population of the zero state

f_Rabi = 50*10**3       # internal state Rabi frequency: units in Hz
f_secular = 1.6*10**6   # secular frequency: units in Hz
f_detuning = 0          # detuning: units in Hz
LD_param = 0.1          # Lamb-Dicke parameter: dimensionless

heating_rate = 0        # heating rate: units in phonons/sec
T_b = 300               # equilibrium temperature of the phonon heat bath: units in Kelvin

t_tot = 1*1e-4            # total time to simulate: units in sec
dt = 5*1e-9               # simulation time step: units in sec
t_scale = (10**6)*dt    # plot time scale: units in us

# returns the mean number of the bath with respect to the motional energy
def return_bath_mean_n():
    hbar = 6.62607015*10**(-34)
    k_B = 1.380649*10**(-23)
    mean_n = 1/(np.exp(hbar*(2*np.pi*f_secular)/(k_B*T_b)) - 1)
    return mean_n
bath_mean_n = return_bath_mean_n()

# returns the coupling efficiency to the bath 
def return_gamma():
    gamma = heating_rate/bath_mean_n
    return gamma
Gamma = return_gamma()

# returns the motional decay rate
def return_motional_decay_rate(n):
    decay_rate = Gamma*((2*bath_mean_n + 1)*n + bath_mean_n)/2    
    return decay_rate

# returns the motional Rabi frequency for different number state transitions
def return_stationary_motional_f_Rabi(n_f, n_i):
    laguerre_factor_0 = complex(1)
    laguerre_factor_1 = complex(1)
    if n_f == n_i:
        laguerre_factor_0 = np.exp(-(LD_param**2)/2)*special.genlaguerre(n_i, 0)(LD_param**2)
        laguerre_factor_1 = laguerre_factor_0
    elif n_f > n_i:
        n_diff = n_f - n_i
        laguerre_factor = np.exp(-(LD_param**2)/2)*(special.factorial(n_i)/special.factorial(n_f))**(1/2)
        laguerre_factor_0 = laguerre_factor*((-LD_param*1j)**n_diff)*special.genlaguerre(n_i, n_diff)(LD_param**2)
        laguerre_factor_1 = laguerre_factor*((LD_param*1j)**n_diff)*special.genlaguerre(n_i, n_diff)(LD_param**2)
    elif n_f < n_i:
        n_diff = n_i - n_f
        laguerre_factor = np.exp(-(LD_param**2)/2)*(special.factorial(n_f)/special.factorial(n_i))**(1/2)
        laguerre_factor_0 = laguerre_factor*((-LD_param*1j)**n_diff)*special.genlaguerre(n_f, n_diff)(LD_param**2)
        laguerre_factor_1 = laguerre_factor*((LD_param*1j)**n_diff)*special.genlaguerre(n_f, n_diff)(LD_param**2)

    return [laguerre_factor_0*(2*np.pi*f_Rabi), laguerre_factor_1*(2*np.pi*f_Rabi)]

modified_f_Rabi = list()
for m in range(n_tot):
    f_Rabi_row = list()
    for n in range(n_tot):
        f_Rabi_row.append(return_stationary_motional_f_Rabi(m, n))
    modified_f_Rabi.append(f_Rabi_row)

def return_evolving_motional_f_Rabi(n_f, n_i, time):
    base_rabi = modified_f_Rabi[n_f][n_i]
    time_factor_0 = np.exp(1j*2*np.pi*((n_f - n_i)*f_secular + f_detuning)*time)
    time_factor_1 = np.exp(1j*2*np.pi*((n_f - n_i)*f_secular - f_detuning)*time)

    return [base_rabi[0]*time_factor_0, base_rabi[1]*time_factor_1]

# initialize the density matrix elements
sim_init_array = list()
sim_empty_array = list()

sim_tls = np.array([[complex(tls_zero_init), complex(0)], [complex(0), complex(1 - tls_zero_init)]])
sim_tls.reshape(2, 2)
sim_empty = np.array([[complex(0), complex(0)], [complex(0), complex(0)]])
sim_empty.reshape(2, 2)

for m in range(n_tot):
    sim_array_row = list()
    sim_empty_row = list()
    for n in range(n_tot):
        prob_n = 0
        if m == n:
            prob_n = (1/(n_mean_init + 1))*(n_mean_init/(n_mean_init + 1))**n
        sim_array_row.append(prob_n*sim_tls)
        sim_empty_row.append(sim_empty)
    sim_init_array.append(sim_array_row)
    sim_empty_array.append(sim_empty_row)
sim_init_array = np.array(sim_init_array)
sim_init_array.reshape(2*n_tot, 2*n_tot)
sim_empty_array = np.array(sim_empty_array)
sim_empty_array.reshape(2*n_tot, 2*n_tot)

# update density matrix elements
sim_timeline = {'time_step' : list(), 'block-diagonal' : list()}

I = np.array([[complex(1), complex(0)], [complex(0), complex(1)]])

def update_K(K_list):
    time_init = time.time()
    for iter in range(round(t_tot/dt)): 
        temp_K_array = list()
        temp_Kd_array = list()
        for m in range(n_tot):
            K_row = list()
            Kd_row = list()
            for n in range(n_tot):
                rabi = return_evolving_motional_f_Rabi(m, n, iter*dt)
                K = -(1j/2)*np.array([[complex(0), rabi[0]], [rabi[1], complex(0)]])
                Kd = (1j/2)*np.array([[complex(0), rabi[0]], [rabi[1], complex(0)]])
                if n == m:
                    decay = return_motional_decay_rate(n)*I
                    K -= decay
                    Kd -= decay
                K_row.append(K)
                Kd_row.append(Kd)
            temp_K_array.append(K_row)
            temp_Kd_array.append(Kd_row)
        temp_K_array = np.array(temp_K_array)
        temp_Kd_array = np.array(temp_Kd_array)
        temp_K_array.reshape(2*n_tot, 2*n_tot)
        temp_Kd_array.reshape(2*n_tot, 2*n_tot)
        K_list.append([temp_K_array, temp_Kd_array])

    print('Diffusion operator update elapsed time: {} sec'.format(round(time.time() - time_init, 1)))
        
def evolve_density(K_list):
    iter_final = round(t_tot/dt)
    iter_curr = 0
    
    sim_prev = cp.deepcopy(sim_init_array)
    sim_next = cp.deepcopy(sim_empty_array)

    time_init = time.time()
    while iter_curr < iter_final:
        # print(iter_curr)

        K_curr = None
        if len(K_list) != 0:
            K_curr = cp.deepcopy(K_list[0])
            K_list.pop()

        if K_curr != None:
            sim_prev_up = shift_array_up(cp.deepcopy(sim_prev))
            sim_prev_down = shift_array_down(cp.deepcopy(sim_prev))

            for m in range(n_tot):
                for n in range(n_tot):
                    sim_prev_up[m][n] = Gamma*(bath_mean_n + 1)*(((m + 1)*(n + 1))**(1/2))*sim_prev_up[m][n]
                    sim_prev_down[m][n] = Gamma*bath_mean_n*((m*n)**(1/2))*sim_prev_down[m][n]
            
            sim_next += sim_prev + (np.einsum('mril, rnlj -> mnij', K_curr[0], sim_prev) + np.einsum('mril, rnlj -> mnij', sim_prev, K_curr[1]) + sim_prev_up + sim_prev_down)*dt

            block_diag = list()
            for diag_n in range(n_tot):
                block_diag.append(sim_next[diag_n][diag_n])
            
            sim_timeline['time_step'].append(iter_curr*t_scale)      # plot units in us
            sim_timeline['block-diagonal'].append(block_diag)

            sim_prev = cp.deepcopy(sim_next)
            sim_next = cp.deepcopy(sim_empty_array)
            iter_curr += 1    
    print('Density matrix evolution elapsed time: {} sec'.format(round(time.time() - time_init, 1)))

    plot_results(sim_timeline)

def shift_array_up(array):
    new_array = np.roll(array, -1 , axis=0)
    new_array = np.roll(new_array, -1 , axis=1)
    new_array[-1] = np.zeros((new_array[-1].shape))
    new_array[:,-1] = np.zeros((new_array[:,-1].shape))
    return new_array

def shift_array_down(array):
    new_array = np.roll(array, 1 , axis=0)
    new_array = np.roll(new_array, 1 , axis=1)
    new_array[0] = np.zeros((new_array[0].shape))
    new_array[:,0] = np.zeros((new_array[:,0].shape))
    return new_array

def plot_results(timeline=dict()):
    prob_one = []
    for data in timeline['block-diagonal']:
        traced = np.array([[complex(0), complex(0)], [complex(0), complex(0)]])
        for diag_n in range(n_tot):
            traced += data[diag_n]
        prob_one.append(traced[1][1].real)

    plt.rcParams['figure.figsize'] = (25,6)
    plt.grid()
    plt.xlim(0, timeline['time_step'][-1])
    plt.ylim(-0.1, 1.1)
    plt.plot(timeline['time_step'], prob_one)
    plt.show()

if __name__ == '__main__':
    print('bath mean number: ', round(bath_mean_n))
    print('Gamma: ', round(Gamma, 6))

    K_manager = multiprocessing.Manager()
    K_list = K_manager.list()

    p1 = multiprocessing.Process(target=update_K, args=[K_list])
    p2 = multiprocessing.Process(target=evolve_density, args=[K_list])

    p1.start()
    p2.start()

    p1.join()
    p2.join()