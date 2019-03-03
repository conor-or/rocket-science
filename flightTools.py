import numpy as np
from numpy import array as A
from matplotlib.pyplot import subplots
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.animation import FuncAnimation
from numpy.linalg import norm

class Rocket:

    def __init__(self):
        """
        Holds all the constant attributes of the rocket
        Units for each quantity are in the square brackets
        """
        self.hull_mass = 1000                       # Mass of the rocket w/o fuel [kg]
        self.fuel_mass = 5000                       # Initial mass of the fuel only [kg]
        self.width = 2                              # Width [m]
        self.height = 20                            # Height [m]
        self.impact_velocity = 3.0                  # Speed above which rocket crashes on impact [m/s]
        self.exhaust_velocity = A([0, 200 * 9.81])  # Specific impulse of engine [s]
        self.max_thrust = 100000                    # Maximum thust of engine [N]


class Flight:

    def __init__(self, flight_controller=None, verbose=True):
        """
        Keeps track of the data and runs the simulation for a single flight
        """

        #  Simulation settings
        self.simulation_resolution = 0.1            # Size of time steps [s] (0.03=30fps for nice animations)
        self.max_runtime = 10.0                     # Maximum allowed simulation time [s]
        self.gravitational_field = A([0.0, -9.81])  # Vector acc. due to gravity [m/s^2]
        self.verbose = verbose                      # Simulation verbosity True/False for printing updates on/off

        # Initialise the rocket
        self.rocket = Rocket()

        # Initial conditions
        # (all of these arrays will be appended to as the simulation runs)
        self.status = A([1]).astype('int')                  # Status (0=Crashed, 1=Flying, 2=Landed)
        self.angle = A([0.0])                               # Angle of thrust (zero points to ground) [rads]
        self.time = A([0.0])                                # Time [s]
        self.position = A([[0.0, 100]])                     # Position vector [m]
        self.velocity = A([[0.0, 0.0]])                     # Velocity vector [m/s]
        self.acceleration = A([self.gravitational_field])   # Acceleration vector [m/s^2]
        self.mass = A([self.rocket.hull_mass +              # Total mass of rocket [kg]
                       self.rocket.fuel_mass])
        self.throttle = A([0.0])                            # Throttle position (from 0 to 1, 1 being full power)

        # Check the flight controller function
        if callable(flight_controller):
            self.flight_controller = flight_controller
        else:
            self.flight_controller = template_controller

    def run(self):
        """
        Runs the simulation given this flight's initial conditions
        and flight controller
        """

        i = 1
        # Start at time step 1 and run until max_runtime or the rocket lands/crashes
        while (self.status[-1] == 1) and (self.time[i - 1] < self.max_runtime):

            # Get the throttle position
            throttle = self.flight_controller(self)

            # If rocket is out of fuel, cut the throttle
            if self.mass[i - 1] <= self.rocket.hull_mass:
                throttle = 0.0

            # Update the flight based on the throttle chosen by the controller
            self.update(throttle)

            # Print the current status
            if self.verbose:
                update_text = 'T: {:05.2f} | {:<6}'.format(self.time[i - 1] + self.simulation_resolution,
                                                           self.status_string())
                print('\r', update_text, end='')

            # Iterate
            i += 1

    def update(self, throttle):
        """
        Updates the position, velocity and mass of the rocket at each
        timestep, given the previous state and the current throttle setting
        """

        # Set delta t for convenience
        dt = self.simulation_resolution

        # Mass expulsion needed to achieve the specified thrust
        delta_m = (throttle * self.rocket.max_thrust * dt) / self.rocket.exhaust_velocity[1]

        # Update the total mass
        self.mass = np.append(self.mass, [self.mass[-1] - delta_m])

        # Update the throttle
        self.throttle = np.append(self.throttle, [throttle])

        # Update the acceleration based on the mass expulsion above
        delta_v = self.rocket.exhaust_velocity * np.log(self.mass[-2] / self.mass[-1])
        total_a = (delta_v / dt) + self.gravitational_field
        self.acceleration = np.append(self.acceleration, [total_a], axis=0)

        # Update the velocity, position and time
        self.velocity = np.append(self.velocity, [self.velocity[-1] + total_a * dt], axis=0)
        self.position = np.append(self.position, [self.position[-1] + self.velocity[-1] * dt], axis=0)
        self.time = np.append(self.time, [self.time[-1] + dt])

        # Check if rocket has landed safely
        if ((self.position[-1][1] <= 0.0) and
            (norm(self.velocity[-1]) <= self.rocket.impact_velocity)):

            self.status = np.append(self.status, [2])
            self.position[-1] = A([self.position[-1][0], 0])[np.newaxis, :]

        # Check if rocket has crashed
        elif self.position[-1][1] <= 0.0:
            self.status = np.append(self.status, [0])
            self.position[-1] = A([self.position[-1][0], 0])[np.newaxis, :]

        else:
            self.status = np.append(self.status, [1])

    def status_string(self):
        """
        Returns the current status of the rocket, given a code 0, 1 or 2
        """
        j = self.status[-1]
        if j == 0:
            ss = 'Crashed'
        elif j == 1:
            ss = 'Flying'
        else:
            ss = 'Landed'
        return ss


class FlightAnimation:

    def __init__(self, flight, filename):
        """
        Animates a given completed flight, saves at filename
        """
        self.flight = flight
        self.animate(filename)

    def animate(self, filename):

        # Get the position for convenience
        pos_x = self.flight.position[0][0]
        pos_y = self.flight.position[0][1]

        # Set an appropriate y-scale
        frame_scale = (self.flight.position[:, 1].max() + self.flight.rocket.height) / 2

        # Initialise figure
        fig, ax = subplots(figsize=(8, 8))
        ax.set(xlim=[- frame_scale, frame_scale], ylim=[0, 2 * frame_scale],
               ylabel='Altitude (m)', xticks=[])

        # Draw the rocket as a rectangle
        self.rocket = Rectangle((pos_x - self.flight.rocket.width / 2, pos_y),
                                self.flight.rocket.width,
                                self.flight.rocket.height,
                                angle=0.0, color='C0', zorder=2)

        # Draw the thrust as an ellipse
        self.thruster = Ellipse((pos_x, pos_y),
                                self.flight.rocket.width,
                                self.flight.rocket.height * self.flight.throttle[0],
                                color='C1', zorder=1)

        # Add these to the plot
        ax.add_artist(self.rocket)
        ax.add_artist(self.thruster)
        fig.tight_layout()

        # Animate the plot according to teh update function (below)
        movie = FuncAnimation(fig, self.update_animation,
                              interval=(1000 * self.flight.simulation_resolution),
                              frames=len(self.flight.position))
        movie.save(filename)

    def update_animation(self, i):

        pos_x = self.flight.position[i][0]
        pos_y = self.flight.position[i][1]

        # Move rocket and thruster to current position
        self.rocket.update({'xy': (pos_x - self.flight.rocket.width / 2, pos_y)})
        self.thruster.center = (pos_x, pos_y)
        self.thruster.height = self.flight.rocket.height * self.flight.throttle[i]


def template_controller(flight):
    """
    This is a template flight control function that performs
    a suicide burn for the default initial conditions
    """

    if 3.0 < flight.position[-1][1] < 58.4:
        throttle = 1.0
    else:
        throttle = 0.0

    return throttle


def flight_data_plot(flight, save=''):
    """
    Plots various data for a given flight
    """

    fig, ax = subplots(5, 1)

    labels = ['Position (m)', 'Velocity (ms$^{-1}$)', 'Acceleration (ms$^{-2}$)',
              'Fuel Used (%)', 'Thrust (%)']

    y_axis = [flight.position[:, 1], flight.velocity[:, 1],
              flight.acceleration[:, 1],
              100.0 * (flight.mass - flight.rocket.hull_mass) / flight.rocket.fuel_mass,
              100.0 * flight.throttle]

    for i, a in enumerate(ax):
        a.plot(flight.time, y_axis[i], color=('C%d' % i))
        a.set_ylabel(labels[i])
        a.set(xlim=[0, flight.time.max()])
        if i < 4:
            a.set_xticks([])
    ax[4].set_xlabel('Time (s)')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)
    fig.set_size_inches(10, 10)

    if save:
        fig.savefig(save)

    return fig
