import numpy as np

class Param:
    def __init__(self, xmax=10.0, res=512, dt=0.05, timesteps=1000, im_time=False):
        self.xmax = xmax
        self.res = res
        self.dt = dt
        self.timesteps = timesteps
        self.dx = 2 * xmax / res
        self.x = np.arange(-xmax + xmax / res, xmax + xmax / res, 2 * xmax / res)
        self.k = (np.arange(0, res // 2) + np.arange(-res // 2, 0)) * np.pi / xmax
        self.im_time = im_time

class Operators:
    def __init__(self, res):
        self.V = np.zeros(res, dtype=complex)
        self.R = np.zeros(res, dtype=complex)
        self.K = np.zeros(res, dtype=complex)
        self.wfc = np.zeros(res, dtype=complex)

def init(par, voffset, wfcoffset):
    opr = Operators(len(par.x))
    opr.V = 0.5 * (par.x - voffset)**2
    opr.wfc = np.exp(-(par.x - wfcoffset)**2 / 2)
    
    if par.im_time:
        opr.K = np.exp(-0.5 * par.k**2 * par.dt)
        opr.R = np.exp(-0.5 * opr.V * par.dt)
    else:
        opr.K = np.exp(-1j * 0.5 * par.k**2 * par.dt)
        opr.R = np.exp(-1j * 0.5 * opr.V * par.dt)
    
    return opr

def split_op(par, opr):
    for i in range(par.timesteps):
        # Half-step in real space
        opr.wfc *= opr.R

        # FFT to momentum space
        opr.wfc = np.fft.fft(opr.wfc)

        # Full step in momentum space
        opr.wfc *= opr.K

        # Inverse FFT back to real space
        opr.wfc = np.fft.ifft(opr.wfc)

        # Final half-step in real space
        opr.wfc *= opr.R

        # Density for plotting and potential
        density = np.abs(opr.wfc) ** 2

        # Renormalizing for imaginary time
        if par.im_time:
            renorm_factor = np.sum(density) * par.dx
            opr.wfc /= np.sqrt(renorm_factor)

        # Outputting data to file
        if i % (par.timesteps // 100) == 0:
            filename = f"output{i:05d}.dat"
            with open(filename, "w") as outfile:
                for j in range(len(density)):
                    outfile.write(f"{par.x[j]}\t{density[j]}\t{opr.V[j].real}\n")
            print(f"Outputting step: {i}")

def calculate_energy(par, opr):
    # Creating real, momentum, and conjugate wavefunctions
    wfc_r = opr.wfc
    wfc_k = np.fft.fft(wfc_r)
    wfc_c = np.conj(wfc_r)

    # Finding the momentum and real-space energy terms
    energy_k = 0.5 * wfc_c * np.fft.ifft((par.k**2) * wfc_k)
    energy_r = wfc_c * opr.V * wfc_r

    # Integrating over all space
    energy_final = np.sum(energy_k + energy_r).real * par.dx

    return energy_final

def main():
    par = Param(5.0, 256, 0.05, 100, True)

    # Starting wavefunction slightly offset so we can see it change
    opr = init(par, 0.0, -1.00)
    split_op(par, opr)

    energy = calculate_energy(par, opr)
    print(f"Energy is: {energy}")

if __name__ == "__main__":
    main()



# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# def initialize_parameters():
#     xmax = 10.0             # Maximum position
#     res = 512               # Resolution (number of grid points)
#     dt = 0.05               # Time step
#     timesteps = 1000        # Number of time steps
#     im_time = True          # Use imaginary time evolution
#     return xmax, res, dt, timesteps, im_time

# # Create position and momentum grids
# def create_grids(xmax, res):
#     dx = 2 * xmax / res
#     x = np.arange(-xmax + xmax / res, xmax + xmax / res, 2 * xmax / res)
#     k = (np.arange(0, res // 2) + np.arange(-res // 2, 0)) * np.pi / xmax
#     return dx, x, k

# # Initialize wavefunction and operators
# def initialize_wavefunction(x):
#     wfc = np.exp(-x**2 / 2) / np.sqrt(np.pi)  # Initial Gaussian wavefunction
#     V = 0.5 * x**2  # Harmonic oscillator potential
#     return wfc, V

# def initialize_operators(k, V, dt, im_time):
#     if im_time:
#         K = np.exp(-0.5 * k**2 * dt)
#         R = np.exp(-0.5 * V * dt)
#     else:
#         K = np.exp(-1j * 0.5 * k**2 * dt)
#         R = np.exp(-1j * 0.5 * V * dt)
#     return K, R

# # Split-operator time evolution
# def split_operator_dynamics(wfc, K, R, timesteps, im_time, dx):
#     density_history = []

#     for i in range(timesteps):
#         # Half-step in real space
#         wfc *= R

#         # FFT to momentum space
#         wfc = np.fft.fft(wfc)

#         # Full step in momentum space
#         wfc *= K

#         # Inverse FFT back to real space
#         wfc = np.fft.ifft(wfc)

#         # Final half-step in real space
#         wfc *= R

#         # Density for plotting and potential
#         density = np.abs(wfc)**2

#         # Renormalize for imaginary time
#         if im_time:
#             renorm_factor = np.sum(density) * dx
#             wfc /= np.sqrt(renorm_factor)

#         density_history.append(density)

#     return density_history

# # Main function
# def main():
#     xmax, res, dt, timesteps, im_time = initialize_parameters()
#     dx, x, k = create_grids(xmax, res)
#     wfc, V = initialize_wavefunction(x)
#     K, R = initialize_operators(k, V, dt, im_time)

#     density_history = split_operator_dynamics(wfc, K, R, timesteps, im_time, dx)

#     # Plot the final wavefunction and potential
#     plt.figure(figsize=(8, 6))
#     plt.plot(x, density_history[-1], label="Wavefunction")
#     plt.plot(x, V, label="Potential Energy")
#     plt.xlabel("Position (x)")
#     plt.ylabel("Density/Potential")
#     plt.legend()
#     plt.title("Final Wavefunction and Potential Energy")
#     plt.show()

# main()