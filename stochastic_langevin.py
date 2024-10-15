import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


# Define the external driving force with noise
def F(noise_f_temp, t):
    """
    Computes the external force with noise.

    Parameters:
    noise_f_temp (float): Random noise component for the force.
    t (float): Current time.

    Returns:
    float: The computed force value with noise added.
    """
    return F0 * np.cos(omega_f * t + noisiness_f * noise_f_temp)


# Define the system of differential equations for the oscillator
def f(noise_f_temp, noise_temp, t, S):
    """
    Defines the system of differential equations for the oscillator.

    Parameters:
    noise_f_temp (float): Noise affecting the driving force.
    noise_temp (float): Noise affecting the system's state.
    t (float): Time variable.
    S (numpy.ndarray): State vector [position, velocity].

    Returns:
    numpy.ndarray: Derivatives of the state vector.
    """
    dSdt = np.zeros_like(S)
    dSdt[0] = S[1]  # Velocity
    dSdt[1] = F(noise_f_temp, t) / m - 2 * r * S[1] - (omega ** 2) * S[0] - noisiness * noise_temp  # Acceleration
    return dSdt


# RK45 solver for solving the system of differential equations
def RK45(f, t0, tf, S0, h):
    """
    Implements the Runge-Kutta 45 method to solve a system of differential equations.

    Parameters:
    f (function): The system of differential equations to solve.
    t0 (float): Initial time.
    tf (float): Final time.
    S0 (array): Initial state vector.
    h (float): Initial step size.

    Returns:
    tuple: Arrays of time values, state vector values, and noise values.
    """
    tau_values = np.array([t0])
    x_values = np.array([[S0[0], S0[1]]])
    t = t0
    noise_iso = np.empty(0, dtype=float)
    noise_f_iso = np.empty(0, dtype=float)

    global cycle_count
    cycle_count = 0
    noise_f_temp = 0

    # Time stepping loop
    while t < tf:
        # Generate noise for each step
        noise_temp = np.random.normal(loc=0, scale=1)
        noise_iso = np.append(noise_iso, noise_temp)

        # Random phase shift after every 3 cycles
        if cycle_count >= 3:
            noise_f_temp += np.random.normal(loc=0, scale=1)
            cycle_count = 0  # Reset cycle count
            noise_f_iso = np.append(noise_f_iso, noisiness_f * noise_f_temp)

        x = x_values[-1, :]

        # Runge-Kutta 45 method calculations
        k1 = h * f(noise_f_temp, noise_temp, t, x)
        k2 = h * f(noise_f_temp, noise_temp, t + (1 / 4) * h, x + (1 / 4) * k1)
        k3 = h * f(noise_f_temp, noise_temp, t + (3 / 8) * h, x + (3 / 32) * k1 + (9 / 32) * k2)
        k4 = h * f(noise_f_temp, noise_temp, t + (12 / 13) * h,
                   x + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)
        k5 = h * f(noise_f_temp, noise_temp, t + h,
                   x + (439 / 216) * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)
        k6 = h * f(noise_f_temp, noise_temp, t + (1 / 2) * h,
                   x - (8 / 27) * k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)

        x_new = x + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4101) * k4 - (1 / 5) * k5
        z_new = x + (16 / 135) * k1 + (6656 / 12825) * k3 + (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6
        error = abs(z_new[0] - x_new[0])
        s = 0.84 * (error_m / error) ** (1 / 4)
        print(f"Current time: {t:.4f}, Step size: {h:.6f}")

        # Error control and adaptive step sizing
        while (error > error_m):
            h = s * h
            k1 = h * f(noise_f_temp, noise_temp, t, x)
            k2 = h * f(noise_f_temp, noise_temp, t + (1 / 4) * h, x + (1 / 4) * k1)
            k3 = h * f(noise_f_temp, noise_temp, t + (3 / 8) * h, x + (3 / 32) * k1 + (9 / 32) * k2)
            k4 = h * f(noise_f_temp, noise_temp, t + (12 / 13) * h,
                       x + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)
            k5 = h * f(noise_f_temp, noise_temp, t + h,
                       x + (439 / 216) * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)
            k6 = h * f(noise_f_temp, noise_temp, t + (1 / 2) * h,
                       x - (8 / 27) * k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)
            x_new = x + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4101) * k4 - (1 / 5) * k5
            z_new = x + (16 / 135) * k1 + (6656 / 12825) * k3 + (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6
            error = abs(z_new[0] - x_new[0])
            s = (error_m / error) ** (1 / 5)
            print(f"Adjusted step size: {h:.6f}, Error: {error:.6e}")

        # Update values
        x_values = np.concatenate((x_values, [x_new]), axis=0)
        tau_values = np.append(tau_values, t + h)
        t += h

        # Update cycle count for driving force noise
        cycle_count += (omega_f * h) / (2 * np.pi)

    return tau_values, x_values, noise_iso


# Envelope function for fitting the exponential decay
def exponential(t, A, gamma):
    """
    Defines the envelope function for fitting the exponential decay of the oscillation.

    Parameters:
    t (float): Time.
    A (float): Amplitude.
    gamma (float): Damping coefficient (inverse of relaxation time T1).

    Returns:
    float: Exponential decay of the envelope.
    """
    return A * np.exp(-gamma * t)


# Global parameters for the system
global r, omega, error_m, omega_f, F0, h_interpolate, noisiness, noisiness_f

# Define system parameters
m = 0.1  # Mass
k = 1000.0  # Spring constant
gamma = 0.5  # Damping coefficient

r = gamma / (2 * m)  # Damping ratio
omega = (k / m) ** 0.5  # Natural frequency

# Allow user to input detuning factor for driving frequency
detuning_factor = float(input("Enter the detuning factor (e.g., 0.5 for half a linewidth): "))

# Initial conditions and other parameters
x0 = 0.0  # Initial position
v0 = 3.0  # Initial velocity
t0 = 0.0  # Start time
tf = 150.0  # End time
h = 0.1  # Step size
h_interpolate = 0.0001  # Step size for interpolation
S0 = np.array([x0, v0])  # Initial state vector
error_m = 1e-6  # Tolerance for error in adaptive step size
F0 = 0.01  # Amplitude of driving force
noisiness = 0  # Noise in the system state
noisiness_f = 0  # Noise in the driving force

linewidth = (gamma / m) / (2 * np.pi)  # Linewidth for underdamped system
omega_f = omega + 2 * np.pi * detuning_factor * linewidth  # Driving frequency with detuning

# Run the RK45 solver without noise in the force
t_values_temp, x_values_temp, noise_iso = RK45(f, t0, tf, S0, h)

# Interpolation for smoother plotting
t_values_spline = np.arange(t0, tf, h_interpolate)
interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
x_values_spline_noiseless = interpolator(t_values_spline)
interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
v_values_spline_noiseless = interpolator(t_values_spline)

# Fourier Transform (for frequency analysis) on the interpolated position values
freqs_noiseless = np.fft.fftshift(np.fft.fftfreq(len(t_values_spline), d=h_interpolate))
X = np.fft.fftshift(np.fft.fft(x_values_spline_noiseless)) * h_interpolate
X_noiseless = np.abs(X)

# Re-run the RK45 solver with noise in the force
noisiness_f = 0.5
t_values_temp, x_values_temp, noise_iso = RK45(f, t0, tf, S0, h)

# Interpolation for smooth plotting
t_values_spline = np.arange(t0, tf, h_interpolate)
interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
x_values_spline_noisy = interpolator(t_values_spline)
interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
v_values_spline_noisy = interpolator(t_values_spline)

# Perform Fourier Transform on the interpolated position values
freqs_noisy = np.fft.fftshift(np.fft.fftfreq(len(t_values_spline), d=h_interpolate))
X = np.fft.fftshift(np.fft.fft(x_values_spline_noisy)) * h_interpolate
X_noisy = np.abs(X)

# Calculate the relaxation time T1
abs_position = np.abs(x_values_spline_noiseless)
initial_guess = [np.max(abs_position), 1.0]
popt, pcov = curve_fit(exponential, t_values_spline, abs_position, p0=initial_guess)
A_fitted, gamma_fitted = popt
T1 = 1 / gamma_fitted
print(f"Relaxation time T1: {T1:.4f} seconds")

# Plot position-time graph
plt.plot(t_values_spline, x_values_spline_noisy, label='Noisy')
plt.plot(t_values_spline, x_values_spline_noiseless, label='Noiseless')
plt.xlabel(r'Time(s)', usetex=True)
plt.ylabel(r'Position(m)', usetex=True)
plt.title(f'Position-Time; Detuned to {detuning_factor} Linewidth(s)', usetex=True)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig('Figure_Time.pdf')
plt.show()

# Plot Fourier Transform graph
plt.plot(freqs_noisy, X_noisy, label='Noisy')
plt.plot(freqs_noiseless, X_noiseless, label='Noiseless')
plt.xlabel(r'Frequency(Hz)', usetex=True)
plt.ylabel(r'Amplitude', usetex=True)
plt.title(f'Fourier Transform; Detuned to {detuning_factor} Linewidth(s)', usetex=True)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig('Figure_FFT.pdf')
plt.show()

# Plot phase space diagram (position vs. momentum)
plt.plot(x_values_spline_noisy, m * v_values_spline_noisy, label='Noisy')
plt.plot(x_values_spline_noiseless, m * v_values_spline_noiseless, label='Noiseless')
plt.xlabel(r'Position(m)', usetex=True)
plt.ylabel(r'Momentum', usetex=True)
plt.title(f'Phase Space Diagram; Detuned to {detuning_factor} Linewidth(s)', usetex=True)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig('Figure_PhaseSpace.pdf')
plt.show()

# Plot the results
plt.figure()
plt.plot(t_values_spline, abs_position, 'b-', label='Absolute Position Data')
plt.plot(t_values_spline, exponential(t_values_spline, *popt), 'r--', label='Fitted Exponential Envelope')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Position Time Series with Fitted Exponential Envelope')
plt.legend()
plt.savefig('Figure_ExponentialFitting.pdf')
plt.show()
