import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation

class Sim_Params:
    def __init__(self,
                 xmax: float,
                 res: int,
                 dt: float,
                 timesteps: int,
                 wf_offset: float,
                 V_offset: float) -> None:
        # constants
        self.m = 1
        self.omega = 1
        self.hbar = 1
        # user values
        self.xmax = xmax
        self.res = res
        self.dt = dt
        self.timesteps = timesteps
        self.wf_offset = wf_offset
        self.V_offset = V_offset
        
        

class Simulator:
    def __init__(self,
                 params: Sim_Params) -> None:
        self.params = params
        # calculate step sizes
        x_length = 2 * params.xmax
        self.dx = x_length / params.res
        self.dp = params.hbar * (2 * np.pi / x_length)
        # create the simulation space
        self.__x_space = np.arange(-params.xmax + (params.xmax / params.res), params.xmax, self.dx)
        self.__p_space = np.concatenate((np.arange(0, params.res / 2), np.arange(-params.res / 2, 0))) * self.dp

        self.__V = 0.5 * params.m * params.omega**2 * (self.__x_space - params.V_offset) ** 2
        self.__initialize()
        self.__x_half_step_operator = np.exp(0.5 * (-1j * self.__V * params.dt))
        self.__p_full_step_operator = np.exp(-1j * (self.__p_space ** 2) * params.dt / (2 * params.m) )

    
    def __initialize(self) -> np.ndarray[complex]:
        self.__current_time = 0
        # initilize wfc
        self.wfc = np.exp(-((self.__x_space - self.params.wf_offset) ** 2) / 2, dtype=complex)

    def __probability_density(self):
        return np.abs(self.wfc)**2
    
    def plot_current_state(self):
        fig: Figure
        ax: Axes
        ax2: Axes
        fig, ax = plt.subplots()
        density = self.__probability_density()
        line1, = ax.plot(self.__x_space, density.real, 'b', label='Density')
        # renorm_factor = sum(density) * self.dx
        ax2 = ax.twinx()
        line2, = ax2.plot(self.__x_space, self.__V.real, 'r', label='Potential')
        ax.set_ylabel('Density', color='b')
        ax2.set_ylabel('Potential', color='r')
        ax.set_title(f'Time: {self.__current_time:.2f}')
        return (fig, ax, ax2)

    def split_evolve(self):
        self.wfc *= self.__x_half_step_operator
        self.wfc = np.fft.fft(self.wfc)
        self.wfc *= self.__p_full_step_operator
        self.wfc = np.fft.ifft(self.wfc)
        self.wfc *= self.__x_half_step_operator
        self.__current_time += self.params.dt


    def simulate(self):
        fig, ax, ax2 = self.plot_current_state()
        line1 = ax.get_lines()[0]
        line2 = ax2.get_lines()[0]
        
        def update(i):
            if i == 0:
                return line1, line2
            self.split_evolve()

            # Density for plotting and potential
            density = self.__probability_density()

            # Update the plot data
            line1.set_ydata(density)
            line2.set_ydata(self.__V.real)
            ax.set_title(f'Time: {self.__current_time:.2f}')

            return line1, line2

        ani = FuncAnimation(fig, update, frames=self.params.timesteps + 1, blit=True)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Density', color='b')
        ax2.set_ylabel('Potential', color='r')
        
        plt.close()  # Close the initial plot since it's not needed
        
        return ani
