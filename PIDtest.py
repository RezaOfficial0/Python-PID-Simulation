import numpy as np
import matplotlib.pyplot as plt

class PID:
    def __init__(self, Kp, Ki, Kd):
        """Initialize PID controller with given gains."""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, process_variable, dt):
        """Compute the PID output based on the error, integral, and derivative."""
        # Calculate error
        error = setpoint - process_variable

        # Integral term (accumulation of past errors)
        self.integral += error * dt

        # Derivative term (rate of change of the error)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0

        # PID output
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Save the current error for the next iteration
        self.prev_error = error

        return output

def system_response(control_input, current_value, tau=1.0, dt=0.1):
    """Simulate a simple first-order system response to the control input."""
    # First-order system equation: x(t+dt) = x(t) + (1/tau) * (u(t) - x(t)) * dt
    return current_value + (1 / tau) * (control_input - current_value) * dt

def simulate_pid_system(setpoint, initial_value, Kp, Ki, Kd, dt, time_end):
    """Simulate the system with the PID controller."""
    # Initialize PID controller
    pid = PID(Kp, Ki, Kd)

    # Time vector for simulation
    time_steps = int(time_end / dt)
    time_array = np.linspace(0, time_end, time_steps)

    # Arrays to store the process values, control signals, and errors
    process_values = np.zeros(time_steps)
    control_signals = np.zeros(time_steps)

    # Initial values
    process_value = initial_value
    for i in range(time_steps):
        # Compute the control signal from the PID controller
        control_signal = pid.compute(setpoint, process_value, dt)

        # Update the process value using the system dynamics
        process_value = system_response(control_signal, process_value, tau=1.0, dt=dt)

        # Store values for plotting
        process_values[i] = process_value
        control_signals[i] = control_signal

    return time_array, process_values, control_signals

# Simulation parameters
setpoint = 1.0  # Target setpoint for the system
initial_value = 0.0  # Initial process value (start from 0)
Kp, Ki, Kd = 3.0, 3, 0.5  # PID gains
dt = 0.1  # Time step
time_end = 10  # Total time for simulation in seconds

# Run the PID control simulation
time_array, process_values, control_signals = simulate_pid_system(setpoint, initial_value, Kp, Ki, Kd, dt, time_end)

# Plotting results
plt.figure(figsize=(12, 8))

# Plot Process Value vs Time
plt.subplot(2, 1, 1)
plt.plot(time_array, process_values, label="Process Value", color='b')
plt.axhline(y=setpoint, color='r', linestyle='--', label="Setpoint")
plt.xlabel("Time [s]")
plt.ylabel("Process Value")
plt.title("PID Control System Response")
plt.legend()

# Plot Control Signal vs Time
plt.subplot(2, 1, 2)
plt.plot(time_array, control_signals, label="Control Signal", color='g')
plt.xlabel("Time [s]")
plt.ylabel("Control Signal")
plt.title("PID Control Signal")
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()
