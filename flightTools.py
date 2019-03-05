import numpy as np
import matplotlib.pyplot as plt
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
        self.impact_velocity = 5.0                  # Speed above which rocket crashes on impact [m/s]
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
        leg_h = 0.2 * self.flight.rocket.height

        # Get the centre of the rocket
        cen_x = pos_x
        cen_y = pos_y + self.flight.rocket.height / 2 + leg_h / 2

        # Set an appropriate y-scale
        frame_scale = self.flight.sim_scale / 2

        # Initialise figure
        fig, ax = subplots(figsize=(8, 8), dpi=150)
        ax.set_facecolor('#f7f7f7')
        ax.set(xlim=[- frame_scale, frame_scale], ylim=[-5.0, 2 * frame_scale],
               ylabel='Altitude (m)', xlabel='')

        telemetry = 'Status: {:>7s}\nT Vel:    {:>5.2f}\nH Vel:    {:>5.2f}\nV Vel:    {:>5.2f}\nScore:    {:>5.2f}'.format(
            self.flight.status_string(0),
            norm(self.flight.velocity[0]), self.flight.velocity[0][0], self.flight.velocity[0][1], self.flight.score[0])
        self.t = ax.text(-45.0, 95, telemetry, ha='left', va='top', fontfamily='monospace')

        self.leg_s = np.tan(30.0 * np.pi / 180.0) * leg_h
        self.leg_l = leg_h / np.cos(30.0 * np.pi / 180.0)
        self.l1 = Rectangle((pos_x - self.flight.rocket.width / 2 - 1.1 * self.leg_s, pos_y), 0.1 * leg_h, 1.1 * self.leg_l,
                            angle=-30.0, color='#879ab7', zorder=5)
        self.l2 = Rectangle((pos_x + self.flight.rocket.width / 2 + 0.95 * self.leg_s, pos_y - leg_h * 0.05), 0.1 * leg_h,
                            1.1 * self.leg_l, angle=30.0, color='#879ab7', zorder=5)

        self.rocket = Rectangle((pos_x - self.flight.rocket.width / 2, pos_y + leg_h), self.flight.rocket.width,
                                self.flight.rocket.height - leg_h,
                                angle=0.0, color='#879ab7', zorder=2)
        self.t0 = Ellipse((pos_x, pos_y + leg_h), self.flight.rocket.width * 0.8,
                          self.flight.rocket.height * thrust_parse(0)[0],
                          color='#e28b44', zorder=1)

        booster_width = self.flight.rocket.width * 0.3
        self.b1 = Rectangle((cen_x - self.flight.rocket.width / 2, cen_y - 2.5 * booster_width),
                            booster_width, booster_width * 5, color='#4e596d', zorder=3)
        self.b2 = Rectangle((cen_x + self.flight.rocket.width / 2 - booster_width, cen_y - 2.5 * booster_width),
                            booster_width, booster_width * 5, color='#4e596d', zorder=3)
        self.t1 = Ellipse((cen_x - self.flight.rocket.width / 2, cen_y),
                          self.flight.rocket.width * 2 * thrust_parse(self.flight.throttle[0])[1], booster_width * 2,
                          color='#e28b44')
        self.t2 = Ellipse((cen_x + self.flight.rocket.width / 2, cen_y),
                          self.flight.rocket.width * 2 * thrust_parse(self.flight.throttle[0])[2], booster_width * 2,
                          color='#e28b44')

        self.ground = Rectangle((- frame_scale, -5), 2 * frame_scale, 5, color='#bcbcbc')
        self.base = Rectangle((-self.flight.base_size, - 5), 2 * self.flight.base_size, 5, color='#686868')

        for patch in [self.rocket, self.t0, self.b1, self.b2,
                      self.t1, self.t2, self.l1, self.l2,
                      self.ground, self.base]:
            ax.add_artist(patch)

        fig.tight_layout()

        # Add an extra second to the end of the animation
        extra_frames = int(1.0 / self.flight.simulation_resolution)

        # Animate the plot according to teh update function (below)
        movie = FuncAnimation(fig, self.update_animation,
                              interval=(1000 * self.flight.simulation_resolution),
                              frames=(len(self.flight.position) + extra_frames))
        movie.save(filename)

    def update_animation(self, j):

        if j > (len(self.flight.position) - 1):
            i = len(self.flight.position) - 1
            throttle = 0
        else:
            i = j
            throttle = self.flight.throttle[i]

        pos_x = self.flight.position[i][0]
        pos_y = self.flight.position[i][1]
        leg_h = 0.2 * self.flight.rocket.height
        booster_width = self.flight.rocket.width * 0.3

        # Get the centre of the rocket
        cen_x = pos_x
        cen_y = pos_y + self.flight.rocket.height / 2 + leg_h / 2

        telemetry = 'Status:   {:>7s}\nT Vel:    {:>7.2f}\nH Vel:    {:>7.2f}\nV Vel:    {:>7.2f}\nScore:    {:>7.2f}'.format(
            self.flight.status_string(i),
            norm(self.flight.velocity[i]), self.flight.velocity[i][0], self.flight.velocity[i][1], self.flight.score[:(i+1)].sum())
        self.t.set_text(telemetry)

        leg_s = np.tan(30.0 * np.pi / 180.0) * leg_h
        leg_l = leg_h / np.cos(30.0 * np.pi / 180.0)

        self.rocket.update({'xy': (pos_x - self.flight.rocket.width / 2, pos_y + leg_h)})
        self.t0.center = (pos_x, pos_y + leg_h)
        self.t0.height = 0.8 * self.flight.rocket.height * thrust_parse(throttle)[0]

        self.l1.update({'xy': (pos_x - self.flight.rocket.width / 2 - 1.1 * leg_s, pos_y)})
        self.l2.update({'xy': (pos_x + self.flight.rocket.width / 2 + 0.95 * leg_s, pos_y - leg_h * 0.05)})

        self.b1.update({'xy': (cen_x - self.flight.rocket.width / 2, cen_y - 2.5 * booster_width)})
        self.b2.update({'xy': (cen_x + self.flight.rocket.width / 2 - booster_width, cen_y - 2.5 * booster_width)})

        self.t1.center = (cen_x - self.flight.rocket.width / 2, cen_y)
        self.t1.width = self.flight.rocket.width * 2 * thrust_parse(throttle)[1]

        self.t2.center = (cen_x + self.flight.rocket.width / 2, cen_y)
        self.t2.width = self.flight.rocket.width * 2 * thrust_parse(throttle)[2]


def thrust_parse(j):
    """
    j in binary gives the appropriate thrust selection:
    Translation:
    Input    0 1 2 4 5 6
    Output...
    Main     0 0 0 1 1 1 2^2
    Left     0 0 1 0 0 1 2^1
    Right    0 1 0 0 1 0 2^0
    """
    if j > 2:
        k = j + 1
    else:
        k = j
    return A([x for x in '{0:03b}'.format(k)]).astype(int)


def template_controller(flight):
    """
    Template for a function that decides on the right
    throttle given the current flight data

    This example performs the most fuel efficient safe landing
    for the initial conditions
    """

    d_i = flight.position[0][1]
    m_i = flight.mass[0]
    f_t = flight.rocket.max_thrust
    imp = flight.rocket.exhaust_velocity[1]
    a = np.sqrt(2 * 9.81 * d_i)

    t = (m_i - (m_i / np.exp(a / imp))) / (f_t / imp)
    d = t * (a / 2)

    # Start burn at the calculated height
    if flight.position[-1][1] <= (d + 0.6):
        throttle = 1.0
    else:
        throttle = 0.0

    return throttle


def template_score_calc(flight):
    """
    Template for a function that calculates the score
    given the current status of the flight

    In this example the score is just the negative height
    so it decreases as the rocket gets further from the ground
    """
    return flight.position[-1][1]


def flight_data_plot(flight, save=''):
    """
    Plots various data for a given flight
    """

    plt.style.use('ggplot')
    fig, ax = plt.subplots(6, 1)

    labels = ['Position (m)', 'Velocity (ms$^{-1}$)', 'Acceleration (ms$^{-2}$)',
              'Fuel Used (%)', 'Throttle (%)', 'Score']

    y_axis = [flight.position[:, 1], flight.velocity[:, 1],
              flight.acceleration[:, 1],
              100.0 * (flight.mass - flight.rocket.hull_mass) / flight.rocket.fuel_mass,
              100.0 * flight.throttle, flight.score]

    for i, a in enumerate(ax):
        a.plot(flight.time, y_axis[i], color=('C%d' % i))
        a.set_ylabel(labels[i])
        a.set(xlim=[0, flight.time.max()])
        if i < 5:
            a.set_xticklabels([])
    ax[4].set_xlabel('Time (s)')

    fig.subplots_adjust(hspace=0.05)
    fig.set_size_inches(10, 12)

    if save:
        fig.savefig(save)

    return fig
