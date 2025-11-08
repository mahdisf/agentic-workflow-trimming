# ============================================================================
# CUSTOMIZED SYSTEM MODULE
# ============================================================================
"""
This module provides a framework for defining and managing customizable system dynamics
for equilibrium trimming and analysis. It allows users to add new systems easily by
following a standardized format.

FORMAT FOR DEFINING A SYSTEM:
Each system is defined in the 'systems' dictionary with the following structure:
{
    'system_name': str,  # Unique name of the system (e.g., 'pendulum')
    'params': dict,      # Dictionary of system parameters (e.g., {'m': 1.0, 'l': 1.0, ...})
    'operating_conditions': dict or callable,  # Operating conditions; if callable, it should return a dict
                                               # (e.g., lambda: {'angle': float(input("Desired angle [rad]: ") or "0.0")})
    'bounds': dict,      # Bounds for states and inputs: {'x_min': list, 'x_max': list, 'u_min': list, 'u_max': list}
    'system_f': callable, # Dynamics function f(x, u, params) -> np.array of state derivatives
    'n_x': int,          # Number of state variables
    'n_u': int,          # Number of input variables
    'state_vars': list,  # List of state variable names (for plotting/labels, can use LaTeX)
    'input_vars': list,  # List of input variable names
    'param_vars': list,  # List of parameter names (usually list(params.keys()))
    'initial_guess_func': callable or None,  # Function to compute initial guess and strategy for PlannerAgent
                                             # def initial_guess_func(operating_conditions, params, n_x, n_u, bounds): return initial_guess, strategy
                                             # If None, PlannerAgent uses default logic. For new systems, define this to avoid modifying agents.py.
}

To add a new system (easy step-by-step guide):

1. **Define the dynamics function**:
   - Create a function that computes the state derivatives: `def f_my_system(x, u, params):`
   - `x` is the state vector (numpy array), `u` is the input vector, `params` is a dict of parameters.
   - Return `np.array([dot_x1, dot_x2, ...])` with the derivatives.
   - Example for a simple pendulum:
     ```python
     def f_simple_pendulum(x, u, params):
         theta, dot_theta = x
         torque = u[0]
         m, l, g = params['m'], params['l'], params['g']
         ddot_theta = (-m * g * l * np.sin(theta) + torque) / (m * l**2)
         return np.array([dot_theta, ddot_theta])
     ```

2. **Define an initial guess function** (optional but recommended for better convergence):
   - Create a function: `def initial_guess_my_system(operating_conditions, params, n_x, n_u, bounds):`
   - Return `initial_guess` (np.array of shape (n_x + n_u,)) and `strategy` (string like 'least_squares').
   - Use physics-based guesses based on operating_conditions.
   - Example:
     ```python
     def initial_guess_simple_pendulum(operating_conditions, params, n_x, n_u, bounds):
         desired_angle = operating_conditions.get('angle', 0.0)
         initial_guess = np.array([desired_angle, 0.0, 0.0])  # theta, dot_theta, torque
         strategy = 'least_squares'
         return initial_guess, strategy
     ```

3. **Add to the systems dictionary**:
   - Copy-paste an existing entry and modify it.
   - Required keys: 'system_name', 'params', 'operating_conditions', 'bounds', 'system_f', 'n_x', 'n_u', 'state_vars', 'input_vars', 'param_vars', 'initial_guess_func'
   - 'operating_conditions' can be a dict or lambda for interactive input.
   - Example entry:
     ```python
     'my_system': {
         'system_name': 'my_system',
         'params': {'param1': 1.0, 'param2': 2.0},
         'operating_conditions': {'condition1': 0.0},  # or lambda: {'condition1': float(input("Enter value: "))}
         'bounds': {'x_min': [-10, -10], 'x_max': [10, 10], 'u_min': [-5], 'u_max': [5]},
         'system_f': f_my_system,
         'n_x': 2,
         'n_u': 1,
         'state_vars': ['state1', 'state2'],
         'input_vars': ['input1'],
         'param_vars': ['param1', 'param2'],
         'initial_guess_func': initial_guess_my_system  # or None
     }
     ```

4. **Test your system**:
   - Run the workflow and select "4. Other" to choose your new system.
   - Ensure it converges and produces reasonable results.
   - Adjust parameters or initial guess if needed.

That's it! Your new system will be automatically available for selection without modifying other files.

========================================================================
"""

import numpy as np

# Dynamics function for damped cart-pole system
def f_cart_pole(x, u, params):
    """
    Damped Cart-Pole System Dynamics
    States: x = [x_cart, theta, dot_x_cart, dot_theta]
           (cart position, pole angle, cart velocity, pole angular velocity)
    Input: u = [force] (force applied to cart)
    Dynamics: Nonlinear equations for damped cart-pole system
    """
    x_cart, theta, dot_x_cart, dot_theta = x
    force = u[0]

    # Extract parameters
    m_cart = params['m_cart']  # Mass of cart
    m_pole = params['m_pole']  # Mass of pole
    l = params['l']            # Length of pole
    g = params['g']            # Gravity
    b_cart = params['b_cart']  # Damping coefficient for cart
    b_pole = params['b_pole']  # Damping coefficient for pole

    # Dynamics equations
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Denominator for angular acceleration
    denom = (m_cart + m_pole) * l - m_pole * l * cos_theta**2

    # Cart acceleration
    ddot_x_cart = (m_pole * l * sin_theta * dot_theta**2 +
                   m_pole * g * sin_theta * cos_theta +
                   force - b_cart * dot_x_cart) / (m_cart + m_pole)

    # Pole angular acceleration
    ddot_theta = (-m_pole * l * cos_theta * sin_theta * dot_theta**2 -
                  (m_cart + m_pole) * g * sin_theta +
                  cos_theta * force - b_pole * dot_theta) / denom

    return np.array([dot_x_cart, dot_theta, ddot_x_cart, ddot_theta])


# Dynamics function for damped pendulum system
def f_damped_pendulum(x, u, params):
    """
    Damped Pendulum System Dynamics
    States: x = [theta, dot_theta] (angle, angular velocity)
    Input: u = [torque] (applied torque)
    Dynamics: Nonlinear pendulum with damping
    """
    theta, dot_theta = x
    torque = u[0]

    # Extract parameters
    m = params['m']  # Mass
    l = params['l']  # Length
    g = params['g']  # Gravity
    b = params['b']  # Damping coefficient

    # Dynamics
    ddot_theta = (-m * g * l * np.sin(theta) - b * dot_theta + torque) / (m * l**2)

    return np.array([dot_theta, ddot_theta])


# Dynamics function for simple pendulum system
def f_simple_pendulum(x, u, params):
    """
    Simple Pendulum System Dynamics
    States: x = [theta, dot_theta] (angle, angular velocity)
    Input: u = [torque] (applied torque)
    Dynamics: Nonlinear simple pendulum
    """
    theta, dot_theta = x
    torque = u[0]

    # Extract parameters
    m = params['m']  # Mass
    l = params['l']  # Length
    g = params['g']  # Gravity

    # Dynamics
    ddot_theta = (-m * g * l * np.sin(theta) + torque) / (m * l**2)

    return np.array([dot_theta, ddot_theta])


# Initial guess functions for systems
def initial_guess_cart_pole(operating_conditions, params, n_x, n_u, bounds):
    """
    Initial guess for cart-pole system based on physics.
    Assumes equilibrium at desired angle with zero velocities.
    """
    desired_angle = operating_conditions.get('angle', 0.0)
    initial_guess = np.array([0.0, desired_angle, 0.0, 0.0, 0.0])  # x_cart, theta, dot_x, dot_theta, force
    strategy = 'least_squares'
    return initial_guess, strategy


def initial_guess_damped_pendulum(operating_conditions, params, n_x, n_u, bounds):
    """
    Initial guess for damped pendulum system.
    Assumes equilibrium at desired angle with zero velocity.
    """
    desired_angle = operating_conditions.get('angle', 0.0)
    initial_guess = np.array([desired_angle, 0.0, 0.0])  # theta, dot_theta, torque
    strategy = 'least_squares'
    return initial_guess, strategy


def initial_guess_simple_pendulum(operating_conditions, params, n_x, n_u, bounds):
    """
    Initial guess for simple pendulum system.
    Assumes equilibrium at desired angle with zero velocity.
    """
    desired_angle = operating_conditions.get('angle', 0.0)
    initial_guess = np.array([desired_angle, 0.0, 0.0])  # theta, dot_theta, torque
    strategy = 'least_squares'
    return initial_guess, strategy


# Systems dictionary
systems = {
    'cart_pole': {
        'system_name': 'cart_pole',
        'params': {'m_cart': 1.0, 'm_pole': 0.1, 'l': 0.5, 'g': 9.81, 'b_cart': 1.0, 'b_pole': 0.1},
        'operating_conditions': lambda: {'angle': float(input("Desired pole angle [rad] (default 0): ") or "0.0")},
        'bounds': {
            'x_min': [-10, -np.pi, -10, -10],
            'x_max': [10, np.pi, 10, 10],
            'u_min': [-50],
            'u_max': [50]
        },
        'system_f': f_cart_pole,
        'n_x': 4,
        'n_u': 1,
        'state_vars': ['$x_{cart}$', r'$\theta$', r'$\dot{x}_{cart}$', r'$\dot{\theta}$'],
        'input_vars': ['$F$'],
        'param_vars': ['m_cart', 'm_pole', 'l', 'g', 'b_cart', 'b_pole'],
        'initial_guess_func': initial_guess_cart_pole
    },
    'damped_pendulum': {
        'system_name': 'damped_pendulum',
        'params': {'m': 1.0, 'l': 1.0, 'g': 9.81, 'b': 0.1},
        'operating_conditions': lambda: {'angle': float(input("Desired angle [rad] (default 0): ") or "0.0")},
        'bounds': {
            'x_min': [-np.pi, -10],
            'x_max': [np.pi, 10],
            'u_min': [-10],
            'u_max': [10]
        },
        'system_f': f_damped_pendulum,
        'n_x': 2,
        'n_u': 1,
        'state_vars': [r'$\theta$', r'$\dot{\theta}$'],
        'input_vars': ['$\\tau$'],
        'param_vars': ['m', 'l', 'g', 'b'],
        'initial_guess_func': initial_guess_damped_pendulum
    },
    'simple_pendulum': {
        'system_name': 'simple_pendulum',
        'params': {'m': 1.0, 'l': 1.0, 'g': 9.81},
        'operating_conditions': lambda: {'angle': float(input("Desired angle [rad] (default 0): ") or "0.0")},
        'bounds': {
            'x_min': [-np.pi, -10],
            'x_max': [np.pi, 10],
            'u_min': [-10],
            'u_max': [10]
        },
        'system_f': f_simple_pendulum,
        'n_x': 2,
        'n_u': 1,
        'state_vars': [r'$\theta$', r'$\dot{\theta}$'],
        'input_vars': ['$\\tau$'],
        'param_vars': ['m', 'l', 'g'],
        'initial_guess_func': initial_guess_simple_pendulum
    }
    # Add more systems here, e.g.,
    # 'my_custom_system': { ... }
}


def list_systems():
    """
    Returns a list of available system names.
    """
    return list(systems.keys())


def get_available_systems():
    """
    Alias for list_systems() for compatibility with agents.py.
    """
    return list_systems()


def get_system_config(system_name):
    """
    Retrieves the full system configuration for the given system name.
    Handles callable operating_conditions by calling them.

    Args:
        system_name (str): Name of the system

    Returns:
        dict: System configuration dictionary

    Raises:
        ValueError: If system_name is not found
    """
    if system_name not in systems:
        available = list_systems()
        raise ValueError(f"System '{system_name}' not found. Available systems: {available}")

    config = systems[system_name].copy()

    # Handle operating_conditions: if callable, call it; else use as is
    if callable(config['operating_conditions']):
        config['operating_conditions'] = config['operating_conditions']()

    return config


def interactive_system_selection():
    """
    Interactive function to select and configure a system.
    Prints available systems, lets user choose, and returns the config.

    Returns:
        dict: System configuration dictionary
    """
    available_systems = list_systems()

    if not available_systems:
        raise ValueError("No systems available in the customized_system module.")

    print("\n" + "="*60)
    print("CUSTOMIZED SYSTEM SELECTION")
    print("="*60)
    print("\nAvailable Customized Systems:")
    for i, name in enumerate(available_systems, 1):
        print(f"{i}. {name}")
    print()

    while True:
        try:
            choice = int(input(f"Select system (1-{len(available_systems)}): ").strip())
            if 1 <= choice <= len(available_systems):
                selected_name = available_systems[choice - 1]
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(available_systems)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print(f"\nSelected system: {selected_name}")
    config = get_system_config(selected_name)

    # Add system_name to config for compatibility
    config['system_name'] = selected_name

    return config


# For backward compatibility and direct use
def ParserAgent(system_name):
    """
    Legacy function for compatibility. Use get_system_config instead.
    """
    return get_system_config(system_name)
