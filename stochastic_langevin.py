import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import pandas as pd
import xlsxwriter
import os


# Define the external driving force function.
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


# Define the system of differential equations for the oscillator.
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
    dSdt[0] = S[1]  # Velocity is the derivative of position.
    dSdt[1] = F(noise_f_temp, t) / m - 2 * r * S[1] - (omega ** 2) * S[0] - noisiness * noise_temp
    return dSdt


# RK45 solver implementation with adaptive step sizing.
def RK45(f, t0, tf, S0, h):
    """
    Implements the Runge-Kutta 45 method to solve a system of differential equations.

    Parameters:
        f (function): The system of differential equations.
        t0 (float): Initial time.
        tf (float): Final time.
        S0 (array): Initial state vector.
        h (float): Initial step size.

    Returns:
        tuple: Arrays of time values, state vectors, and noise values.
    """
    tau_values = np.array([t0])
    x_values = np.array([[S0[0], S0[1]]])
    t = t0
    noise_iso = np.empty(0, dtype=float)
    noise_f_iso = np.empty(0, dtype=float)

    global cycle_count
    cycle_count = 0
    noise_f_temp = 0

    # Main time-stepping loop with adaptive step size control.
    while t < tf:
        noise_temp = np.random.normal(loc=0, scale=1)
        noise_iso = np.append(noise_iso, noise_temp)

        # Update the driving force noise every 3 cycles.
        if cycle_count >= 3:
            noise_f_temp += np.random.normal(loc=0, scale=1)
            cycle_count = 0
            noise_f_iso = np.append(noise_f_iso, noisiness_f * noise_f_temp)

        x = x_values[-1, :]

        # Compute RK45 coefficients.
        k1 = h * f(noise_f_temp, noise_temp, t, x)
        k2 = h * f(noise_f_temp, noise_temp, t + (1 / 4) * h, x + (1 / 4) * k1)
        k3 = h * f(noise_f_temp, noise_temp, t + (3 / 8) * h, x + (3 / 32) * k1 + (9 / 32) * k2)
        k4 = h * f(noise_f_temp, noise_temp, t + (12 / 13) * h,
                   x + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)
        k5 = h * f(noise_f_temp, noise_temp, t + h,
                   x + (439 / 216) * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)
        k6 = h * f(noise_f_temp, noise_temp, t + (1 / 2) * h,
                   x - (8 / 27) * k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)

        # Estimate new state with two methods to calculate the local error.
        x_new = x + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4101) * k4 - (1 / 5) * k5
        z_new = x + (16 / 135) * k1 + (6656 / 12825) * k3 + (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6
        error = abs(z_new[0] - x_new[0])
        s = 0.84 * (error_m / error) ** (1 / 4)
        print(f"Current time: {t:.4f}, Step size: {h:.6f}")

        # Adjust the step size until the error is within the specified tolerance.
        while error > error_m:
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

        # Update state and time arrays.
        x_values = np.concatenate((x_values, [x_new]), axis=0)
        tau_values = np.append(tau_values, t + h)
        t += h

        # Increment cycle count based on the driving frequency.
        cycle_count += (omega_f * h) / (2 * np.pi)

    return tau_values, x_values, noise_iso


# Exponential function for envelope fitting.
def exponential(t, A, gamma):
    """
    Defines an exponential decay function for envelope fitting.

    Parameters:
        t (float): Time.
        A (float): Amplitude.
        gamma (float): Damping coefficient.

    Returns:
        float: Exponential decay value.
    """
    return A * np.exp(-gamma * t)


# Global and system parameters.
global r, omega, error_m, omega_f, F0, h_interpolate, noisiness, noisiness_f

# Define physical parameters.
m = 0.1  # Mass
k = 1000.0  # Spring constant
gamma = 0.5  # Damping coefficient

r = gamma / (2 * m)  # Damping ratio
omega = (k / m) ** 0.5  # Natural frequency

# Prompt for a detuning factor to adjust the driving frequency.
detuning_factor = float(input("Enter the detuning factor (e.g., 0.5 for half a linewidth): "))

# Define initial conditions and simulation parameters.
x0 = 0.0  # Initial position
v0 = 3.0  # Initial velocity
t0 = 0.0  # Start time
tf = 150.0  # End time
h = 0.1  # Initial step size for integration
h_interpolate = 0.0001  # Interpolation step size for smooth plots
S0 = np.array([x0, v0])  # Initial state vector [position, velocity]
error_m = 1e-6  # Error tolerance for adaptive integration
F0 = 0.01  # Amplitude of the driving force
noisiness = 1  # Noise factor for the system state
noisiness_f = 1  # Noise factor for the driving force

# Calculate the linewidth and adjust the driving frequency.
linewidth = (gamma / m) / (2 * np.pi)
omega_f = omega + 2 * np.pi * detuning_factor * linewidth

# Run the simulation without noise in the driving force.
t_values_temp, x_values_temp, noise_iso = RK45(f, t0, tf, S0, h)

# Interpolate the noiseless data for smoother plotting.
t_values_spline = np.arange(t0, tf, h_interpolate)
interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
x_values_spline_noiseless = interpolator(t_values_spline)
interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
v_values_spline_noiseless = interpolator(t_values_spline)

# Perform Fourier Transform on the noiseless position data.
freqs_noiseless = np.fft.fftshift(np.fft.fftfreq(len(t_values_spline), d=h_interpolate))
X = np.fft.fftshift(np.fft.fft(x_values_spline_noiseless)) * h_interpolate
X_noiseless = np.abs(X)

# Run the simulation with noise in the driving force.
noisiness_f = 0.5
t_values_temp, x_values_temp, noise_iso = RK45(f, t0, tf, S0, h)

# Interpolate the noisy data for smoother plotting.
t_values_spline = np.arange(t0, tf, h_interpolate)
interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
x_values_spline_noisy = interpolator(t_values_spline)
interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
v_values_spline_noisy = interpolator(t_values_spline)

# Perform Fourier Transform on the noisy position data.
freqs_noisy = np.fft.fftshift(np.fft.fftfreq(len(t_values_spline), d=h_interpolate))
X = np.fft.fftshift(np.fft.fft(x_values_spline_noisy)) * h_interpolate
X_noisy = np.abs(X)

# Fit an exponential envelope to the noiseless absolute position data to estimate T1.
abs_position = np.abs(x_values_spline_noiseless)
initial_guess = [np.max(abs_position), 1.0]
popt, pcov = curve_fit(exponential, t_values_spline, abs_position, p0=initial_guess)
A_fitted, gamma_fitted = popt
T1 = 1 / gamma_fitted
print(f"Relaxation time T1: {T1:.4f} seconds")

# Create directories for PDF plots, PNG images, and data exports.
pdf_dir = "pdf"
png_dir = "png"
data_dir = "data"

os.makedirs(pdf_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# -------------------------
# Export PDF plots (do not display)
# -------------------------

# Position-Time plot
plt.plot(t_values_spline, x_values_spline_noisy, label='Noisy')
plt.plot(t_values_spline, x_values_spline_noiseless, label='Noiseless')
plt.xlabel(r'Time(s)', usetex=True)
plt.ylabel(r'Position(m)', usetex=True)
plt.title(f'Position-Time; Detuned to {detuning_factor} Linewidth(s)', usetex=True)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(pdf_dir, 'Figure_Time.pdf'))
plt.close()  # Do not display the figure

# Fourier Transform plot
plt.plot(freqs_noisy, X_noisy, label='Noisy')
plt.plot(freqs_noiseless, X_noiseless, label='Noiseless')
plt.xlabel(r'Frequency(Hz)', usetex=True)
plt.ylabel(r'Amplitude', usetex=True)
plt.title(f'Fourier Transform; Detuned to {detuning_factor} Linewidth(s)', usetex=True)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(pdf_dir, 'Figure_FFT.pdf'))
plt.close()

# Phase Space plot
plt.plot(x_values_spline_noisy, m * v_values_spline_noisy, label='Noisy')
plt.plot(x_values_spline_noiseless, m * v_values_spline_noiseless, label='Noiseless')
plt.xlabel(r'Position(m)', usetex=True)
plt.ylabel(r'Momentum', usetex=True)
plt.title(f'Phase Space Diagram; Detuned to {detuning_factor} Linewidth(s)', usetex=True)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(pdf_dir, 'Figure_PhaseSpace.pdf'))
plt.close()

# Exponential Fitting plot
plt.figure()
plt.plot(t_values_spline, abs_position, 'b-', label='Absolute Position Data')
plt.plot(t_values_spline, exponential(t_values_spline, *popt), 'r--', label='Fitted Exponential Envelope')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Position Time Series with Fitted Exponential Envelope')
plt.legend()
plt.savefig(os.path.join(pdf_dir, 'Figure_ExponentialFitting.pdf'))
plt.close()

# -------------------------
# Export PNG plots (for Excel insertion)
# -------------------------

# Re-create and export the Position-Time plot as PNG
fig_time = plt.figure()
plt.plot(t_values_spline, x_values_spline_noisy, label='Noisy')
plt.plot(t_values_spline, x_values_spline_noiseless, label='Noiseless')
plt.xlabel(r'Time(s)', usetex=True)
plt.ylabel(r'Position(m)', usetex=True)
plt.title(f'Position-Time; Detuned to {detuning_factor} Linewidth(s)', usetex=True)
plt.grid(True)
plt.tight_layout()
plt.legend()
fig_time.savefig(os.path.join(png_dir, 'Figure_Time.png'))
plt.close(fig_time)

# Re-create and export the Fourier Transform plot as PNG
fig_fft = plt.figure()
plt.plot(freqs_noisy, X_noisy, label='Noisy')
plt.plot(freqs_noiseless, X_noiseless, label='Noiseless')
plt.xlabel(r'Frequency(Hz)', usetex=True)
plt.ylabel(r'Amplitude', usetex=True)
plt.title(f'Fourier Transform; Detuned to {detuning_factor} Linewidth(s)', usetex=True)
plt.grid(True)
plt.tight_layout()
plt.legend()
fig_fft.savefig(os.path.join(png_dir, 'Figure_FFT.png'))
plt.close(fig_fft)

# Re-create and export the Phase Space plot as PNG
fig_phase = plt.figure()
plt.plot(x_values_spline_noisy, m * v_values_spline_noisy, label='Noisy')
plt.plot(x_values_spline_noiseless, m * v_values_spline_noiseless, label='Noiseless')
plt.xlabel(r'Position(m)', usetex=True)
plt.ylabel(r'Momentum', usetex=True)
plt.title(f'Phase Space Diagram; Detuned to {detuning_factor} Linewidth(s)', usetex=True)
plt.grid(True)
plt.tight_layout()
plt.legend()
fig_phase.savefig(os.path.join(png_dir, 'Figure_PhaseSpace.png'))
plt.close(fig_phase)

# Re-create and export the Exponential Fitting plot as PNG
fig_exp = plt.figure()
plt.plot(t_values_spline, abs_position, 'b-', label='Absolute Position Data')
plt.plot(t_values_spline, exponential(t_values_spline, *popt), 'r--', label='Fitted Exponential Envelope')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Position Time Series with Fitted Exponential Envelope')
plt.legend()
fig_exp.savefig(os.path.join(png_dir, 'Figure_ExponentialFitting.png'))
plt.close(fig_exp)

# -------------------------
# Export simulation data and create an Excel workbook.
# -------------------------

# Export the simulation data as a CSV file.
df = pd.DataFrame({
    "Time": t_values_spline,
    "Noisy Position": x_values_spline_noisy,
    "Noiseless Position": x_values_spline_noiseless,
    "Noisy Velocity": v_values_spline_noisy,
    "Noiseless Velocity": v_values_spline_noiseless,
    "Absolute Position": abs_position
})
csv_file = os.path.join(data_dir, 'simulation_data.csv')
df.to_csv(csv_file, index=False)
print(f"Data series exported to {csv_file}")

# Create an Excel workbook with multiple worksheets.
excel_file = os.path.join(data_dir, 'simulation_output.xlsx')
workbook = xlsxwriter.Workbook(excel_file)

worksheet_data = workbook.add_worksheet('Data')
headers = ["Time", "Noisy Position", "Noiseless Position", "Noisy Velocity", "Noiseless Velocity", "Absolute Position"]
for col, header in enumerate(headers):
    worksheet_data.write(0, col, header)
for i, (time_val, pos_noisy, pos_noiseless, vel_noisy, vel_noiseless, abs_pos_val) in enumerate(
        zip(t_values_spline, x_values_spline_noisy, x_values_spline_noiseless, v_values_spline_noisy,
            v_values_spline_noiseless, abs_position),
        start=1):
    worksheet_data.write(i, 0, time_val)
    worksheet_data.write(i, 1, pos_noisy)
    worksheet_data.write(i, 2, pos_noiseless)
    worksheet_data.write(i, 3, vel_noisy)
    worksheet_data.write(i, 4, vel_noiseless)
    worksheet_data.write(i, 5, abs_pos_val)

worksheet_time = workbook.add_worksheet('Time Plot')
worksheet_time.insert_image('B2', os.path.join(png_dir, 'Figure_Time.png'))

worksheet_fft = workbook.add_worksheet('FFT Plot')
worksheet_fft.insert_image('B2', os.path.join(png_dir, 'Figure_FFT.png'))

worksheet_phase = workbook.add_worksheet('Phase Space')
worksheet_phase.insert_image('B2', os.path.join(png_dir, 'Figure_PhaseSpace.png'))

worksheet_exp = workbook.add_worksheet('Exponential Fit')
worksheet_exp.insert_image('B2', os.path.join(png_dir, 'Figure_ExponentialFitting.png'))

workbook.close()
print(f"Excel workbook exported to {excel_file}")
