import numpy as np
from scipy.optimize import minimize, fsolve, least_squares
from scipy.linalg import eigvals
import warnings

# ============================================================================
# SYSTEM DYNAMICS DEFINITIONS
# ============================================================================

def f_msd(x, u, params):
    """
    Mass-Spring-Damper System
    States: x = [position, velocity]
    Input: u = [force]
    Dynamics: m*x_ddot + c*x_dot + k*x = F
    """
    m, c, k = params['m'], params['c'], params['k']
    pos, vel = x[0], x[1]
    force = u[0]

    x_dot = vel
    x_ddot = (force - c * vel - k * pos) / m

    return np.array([x_dot, x_ddot])


def f_heli(x, u, params):
    """
    2-DOF Helicopter System
    States: x = [theta, psi, dot_theta, dot_psi]
           (pitch angle, travel angle, pitch rate, travel rate)
    Inputs: u = [V_p, V_y] (pitch voltage, yaw voltage)
    """
    theta, psi, p, r = x
    V_p, V_y = u

    J_p = params['J_p']
    D_p = params['D_p']
    K_sp = params['K_sp']
    K_pp = params['K_pp']
    K_py = params['K_py']
    J_y = params['J_y']
    D_y = params['D_y']
    K_yp = params['K_yp']
    K_yy = params['K_yy']

    dot_theta = p
    dot_psi = r
    dot_p = (-(K_sp / J_p) * np.sin(theta) - (D_p / J_p) * p +
             (K_pp / J_p) * V_p + (K_py / J_p) * V_y)
    dot_r = (-(D_y / J_y) * r + (K_yp / J_y) * V_p * np.cos(theta) +
             (K_yy / J_y) * V_y * np.cos(theta))

    return np.array([dot_theta, dot_psi, dot_p, dot_r])


def f_aircraft(x, u, params):
    """
    Aircraft Longitudinal Dynamics
    States: x = [V, alpha, q, theta]
           (airspeed, angle of attack, pitch rate, pitch angle)
    Input: u = [delta_e] (elevator deflection)
    """
    V, alpha, q, theta = x
    delta_e = u[0]

    # Extract parameters
    m = params['m']
    g = params['g']
    rho = params['rho']
    S = params['S']
    c = params['c']
    I_yy = params['I_yy']
    T = params['T']

    # Aerodynamic coefficients
    C_L0 = params['C_L0']
    C_Lalpha = params['C_Lalpha']
    C_Lq = params['C_Lq']
    C_Ldelta = params['C_Ldelta']
    C_D0 = params['C_D0']
    k = params['k']
    C_m0 = params['C_m0']
    C_malpha = params['C_malpha']
    C_mq = params['C_mq']
    C_mdelta = params['C_mdelta']

    # Dynamic pressure
    Q = 0.5 * rho * V**2

    # Aerodynamic forces and moments
    C_L = C_L0 + C_Lalpha * alpha + C_Lq * (c / (2 * V)) * q + C_Ldelta * delta_e
    C_D = C_D0 + k * C_L**2
    C_m = C_m0 + C_malpha * alpha + C_mq * (c / (2 * V)) * q + C_mdelta * delta_e

    L = Q * S * C_L
    D = Q * S * C_D
    M = Q * S * c * C_m

    # Flight path angle
    gamma = theta - alpha

    # State derivatives
    dot_V = (T * np.cos(alpha) - D) / m - g * np.sin(gamma)
    dot_alpha = q - (L + T * np.sin(alpha)) / (m * V) + g * np.cos(gamma) / V
    dot_q = M / I_yy
    dot_theta = q

    return np.array([dot_V, dot_alpha, dot_q, dot_theta])


class PlannerAgent:
    """
    Interprets operating conditions and determines solving strategy.
    In production, this would use LLM to parse natural language specifications.
    """
    def __init__(self):
        self.name = "PlannerAgent"

    def act(self, operating_conditions, system_name, params, n_x, n_u, bounds, trace):
        """
        Determines initial guess and strategy based on operating conditions.
        
        Returns:
            initial_guess: np.array of shape (n_x + n_u,)
            strategy: str, optimization method to use
            trace: updated trace list
        """
        initial_guess, strategy = self._get_initial_guess_and_strategy(
            operating_conditions, system_name, params, n_x, n_u, bounds
        )
        
        trace.append({
            "agent": self.name,
            "action": "plan_trimming_strategy",
            "decision": {
                "initial_guess": initial_guess.tolist(),
                "strategy": strategy,
                "reasoning": f"Based on {system_name} operating conditions: {operating_conditions}"
            }
        })
        return initial_guess, strategy, trace

    def _get_initial_guess_and_strategy(self, operating_conditions, system_name, params, n_x, n_u, bounds):
        """
        Rule-based logic for initial guess determination.
        In LLM version, prompt would be: "Given system {system_name} with parameters {params}
        and operating conditions {operating_conditions}, suggest physically reasonable initial guess."
        """
        # First, try to get custom initial guess function from customized_system.py
        try:
            import customized_system
            if hasattr(customized_system, 'systems') and system_name in customized_system.systems:
                custom_func = customized_system.systems[system_name].get('initial_guess_func')
                if custom_func is not None:
                    return custom_func(operating_conditions, params, n_x, n_u, bounds)
        except (ImportError, AttributeError, KeyError):
            # Fall back to built-in logic if custom function not available
            pass

        # Built-in logic for known systems
        if system_name == 'msd':
            # For mass-spring-damper: x_e = F_e/k, v_e = 0
            desired_u = operating_conditions.get('desired_force', 1.0)
            x_pos = desired_u / params['k']
            initial_guess = np.array([x_pos, 0.0, desired_u])
            strategy = 'minimize'

        elif system_name == 'heli':
            # For helicopter hover: all angles and rates zero, voltages balanced
            if operating_conditions.get('mode', 'hover') == 'hover':
                # Balance gravity: V_p_guess = (m*g*l_p) / K_pp
                V_p_guess = (params['m'] * params['g'] * params['l_p']) / params['K_pp']
                initial_guess = np.array([0.0, 0.0, 0.0, 0.0, V_p_guess, 0.0])
            else:
                # Perturbed equilibrium
                theta_tilt = operating_conditions.get('theta_tilt', 0.1)
                V_p_guess = (params['K_sp'] / params['K_pp']) * np.sin(theta_tilt)  # ~34 (rough)
                V_y_guess = - (params['K_yp'] / params['K_yy']) * V_p_guess  # ~42 (rough)
                initial_guess = np.array([theta_tilt, 0.0, 0.0, 0.0, V_p_guess, V_y_guess])
            # Always constrain psi (travel angle) to 0.0 to prevent drift
            strategy = 'least_squares'

        elif system_name == 'aircraft':
            # For level flight: gamma = 0, so theta = alpha
            V_des = operating_conditions.get('airspeed', 50.0)
            # Initial alpha guess from lift requirement: L = mg
            m, g, rho, S = params['m'], params['g'], params['rho'], params['S']
            C_L0, C_Lalpha = params['C_L0'], params['C_Lalpha']
            # Estimate: C_L = mg/(0.5*rho*V^2*S) = C_L0 + C_Lalpha*alpha
            C_L_req = (m * g) / (0.5 * rho * V_des**2 * S)
            alpha_guess = max(0.01, (C_L_req - C_L0) / C_Lalpha)
            alpha_guess = np.clip(alpha_guess, 0.01, 0.15)  # Reasonable range

            # Add to aircraft block
            a = 0.5 * params['rho'] * params['S'] * params['C_D0']
            b = 2 * params['k'] * (params['m'] * params['g'])**2 / (params['rho'] * params['S'])
            # Solve a V^4 - T V^2 + b = 0
            T = params['T']
            disc = T**2 - 4 * a * b
            if disc > 0:
                w1 = (T + np.sqrt(disc)) / (2 * a)
                w2 = (T - np.sqrt(disc)) / (2 * a)
                V_est = np.sqrt(max(w1, w2)) if max(w1, w2) > 0 else V_des  # Take larger valid
                V_des = np.clip(V_est, 20, 100)  # Bounds
            # Then proceed with C_L_req using V_des, and estimate delta_e = -(C_m0 + C_malpha * alpha_guess) / C_mdelta
            delta_e_guess = - (params['C_m0'] + params['C_malpha'] * alpha_guess) / params['C_mdelta']
            initial_guess = np.array([V_des, alpha_guess, 0.0, alpha_guess, delta_e_guess])
            strategy = 'least_squares'
        else:
            raise ValueError(f"Unknown system: {system_name}")

        # Ensure initial guess is within bounds
        initial_guess = self._clip_to_bounds(initial_guess, bounds, n_x, n_u)
        return initial_guess, strategy
    
    def _clip_to_bounds(self, guess, bounds, n_x, n_u):
        """Clip initial guess to specified bounds."""
        x_min, x_max = np.array(bounds['x_min']), np.array(bounds['x_max'])
        u_min, u_max = np.array(bounds['u_min']), np.array(bounds['u_max'])
        
        guess[:n_x] = np.clip(guess[:n_x], x_min, x_max)
        guess[n_x:] = np.clip(guess[n_x:], u_min, u_max)
        return guess


class SolverAgent:
    """
    Calls numerical optimization tools to find equilibrium points.
    Implements retry logic with adaptive strategies.
    """
    def __init__(self, max_retries=25, tolerance=1e-8):
        self.name = "SolverAgent"
        self.max_retries = max_retries
        self.tolerance = tolerance

    def act(self, system_f, initial_guess, params, n_x, n_u, strategy, bounds, trace):
        """
        Iteratively calls solve_equilibrium until convergence.
        
        Returns:
            x_e, u_e: equilibrium state and input
            converged: bool
            trace: updated trace list
        """
        current_guess = initial_guess.copy()
        
        for attempt in range(self.max_retries):
            x_e, u_e, converged, cost, message, trace_entry = solve_equilibrium(
                system_f, current_guess, params, n_x, n_u, strategy, bounds, self.tolerance
            )
            trace.append(trace_entry)

            print(f"Attempt {attempt}: cost={cost}, success={converged}, message={message}")

            if converged:
                trace.append({
                    "agent": self.name,
                    "action": "convergence_achieved",
                    "attempt": attempt + 1,
                    "final_cost": float(cost)
                })
                return x_e, u_e, converged, trace
            
            # Adaptive retry strategy
            if attempt < self.max_retries - 1:
                current_guess = self._adapt_guess(current_guess, x_e, u_e, n_x, n_u, bounds, attempt)
                trace.append({
                    "agent": self.name,
                    "action": "retry_with_adapted_guess",
                    "attempt": attempt + 1,
                    "new_guess": current_guess.tolist(),
                    "previous_cost": float(cost)
                })
        
        # Failed to converge
        trace.append({
            "agent": self.name,
            "action": "convergence_failed",
            "attempts": self.max_retries
        })
        return None, None, False, trace
    
    def _adapt_guess(self, old_guess, x_e, u_e, n_x, n_u, bounds, attempt):
        """
        Adaptive guess modification strategy.
        In LLM version: "Previous attempt yielded cost X. Suggest modified guess."
        """
        # Strategy 1: Use result from failed attempt if it's reasonable
        if x_e is not None and u_e is not None:
            new_guess = np.concatenate([x_e, u_e])
        else:
            new_guess = old_guess.copy()
        
        # Strategy 2: Add random perturbation with decreasing magnitude
        perturbation_scale = 0.1 * (0.5 ** attempt)
        new_guess += perturbation_scale * np.random.randn(len(new_guess))
        
        # Strategy 3: Ensure bounds are respected
        x_min, x_max = np.array(bounds['x_min']), np.array(bounds['x_max'])
        u_min, u_max = np.array(bounds['u_min']), np.array(bounds['u_max'])
        new_guess[:n_x] = np.clip(new_guess[:n_x], x_min, x_max)
        new_guess[n_x:] = np.clip(new_guess[n_x:], u_min, u_max)
        
        return new_guess


class AnalysisAgent:
    """
    Performs linearization, stability analysis, and constraint validation.
    Orchestrates multiple analysis tools in sequence.
    """
    def __init__(self):
        self.name = "AnalysisAgent"

    def act(self, system_f, x_e, u_e, params, bounds, trace):
        """
        Calls analysis tools sequentially: linearization → stability → validation.
        
        Returns:
            A, B: linearized system matrices
            eigenvalues: list of eigenvalues
            classification: stability classification string
            feasible: bool indicating constraint satisfaction
            trace: updated trace list
        """
        trace.append({
            "agent": self.name,
            "action": "begin_analysis",
            "equilibrium": {"x_e": x_e.tolist(), "u_e": u_e.tolist()}
        })
        
        # Tool 1: Linearization
        A, B, trace_entry = compute_jacobian(system_f, x_e, u_e, params)
        trace.append(trace_entry)
        
        # Tool 2: Stability analysis
        eigenvalues, classification, trace_entry = analyze_stability(A)
        trace.append(trace_entry)
        
        # Tool 3: Constraint validation
        feasible, trace_entry = validate_constraints(x_e, u_e, bounds)
        trace.append(trace_entry)
        
        trace.append({
            "agent": self.name,
            "action": "analysis_complete",
            "summary": {
                "stability": classification,
                "feasible": feasible
            }
        })
        
        return A, B, eigenvalues, classification, feasible, trace


# ============================================================================
# TOOLS: Specialized functions called by agents
# ============================================================================

def solve_equilibrium(system_f, initial_guess, params, n_x, n_u, strategy, bounds, tolerance=1e-10):
    """
    Tool: Find equilibrium point where f(x_e, u_e) = 0.
    
    Args:
        system_f: dynamics function f(x, u, params)
        initial_guess: np.array of shape (n_x + n_u,)
        params: dict of system parameters
        n_x: number of states
        n_u: number of inputs
        strategy: 'minimize' or 'fsolve'
        bounds: dict with x_min, x_max, u_min, u_max
        tolerance: convergence tolerance
        
    Returns:
        x_e, u_e: equilibrium state and input
        converged: bool
        cost: final optimization cost
        trace_entry: dict for logging
    """
    def eq_func(z):
        """Equilibrium condition: f(x, u) = 0"""
        x, u = z[:n_x], z[n_x:]
        return system_f(x, u, params)
    
    # Set up bounds for optimization
    x_min, x_max = np.array(bounds['x_min']), np.array(bounds['x_max'])
    u_min, u_max = np.array(bounds['u_min']), np.array(bounds['u_max'])
    opt_bounds = [(x_min[i], x_max[i]) for i in range(n_x)] + \
                 [(u_min[j], u_max[j]) for j in range(n_u)]
    
    # Define Jacobian function for analytical gradients
    def jac_func(z):
        """Numerical Jacobian of eq_func for better optimization performance."""
        eps = 1e-7
        f0 = eq_func(z)
        n_f = len(f0)
        n_z = len(z)
        J = np.zeros((n_f, n_z))
        for j in range(n_z):
            z_pert = z.copy()
            z_pert[j] += eps
            f_pert = eq_func(z_pert)
            J[:, j] = (f_pert - f0) / eps
        return J

    try:
        if strategy == 'minimize':
            # Minimize ||f(x,u)||^2 (works for non-square systems)
            def cost_func(z):
                f_val = eq_func(z)
                return np.sum(f_val**2)

            def cost_jac(z):
                f_val = eq_func(z)
                J = jac_func(z)
                return 2 * f_val @ J

            with warnings.catch_warnings():
                warnings.simplefilter("default")
                res = minimize(cost_func, initial_guess, method='L-BFGS-B',
                             jac=cost_jac, bounds=opt_bounds, options={'ftol': tolerance})

            x_e, u_e = res.x[:n_x], res.x[n_x:]
            cost = res.fun
            converged = res.success and cost < tolerance
            message = res.message

        elif strategy == 'least_squares':
            # Use least_squares for better convergence on nonlinear problems
            with warnings.catch_warnings():
                warnings.simplefilter("default")
                res = least_squares(eq_func, initial_guess, jac='3-point',
                                  bounds=(np.concatenate([x_min, u_min]), np.concatenate([x_max, u_max])),
                                  ftol=tolerance, xtol=tolerance, gtol=tolerance)

            x_e, u_e = res.x[:n_x], res.x[n_x:]
            cost = np.sum(res.fun**2)
            converged = res.success and cost < tolerance
            message = res.message

        elif strategy == 'fsolve':
            # Direct root finding (requires n_x == n_x + n_u or careful setup)
            with warnings.catch_warnings():
                warnings.simplefilter("default")
                sol = fsolve(eq_func, initial_guess, full_output=True)

            x_e, u_e = sol[0][:n_x], sol[0][n_x:]
            info_dict = sol[1]
            if info_dict is not None and 'fvec' in info_dict:
                cost = np.sum(info_dict['fvec']**2)
                converged = info_dict.get('info', 0) in [1, 2, 3, 4] and cost < tolerance
            else:
                cost = np.inf
                converged = False
            message = sol[3] if len(sol) > 3 else "No message"

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    except Exception as e:
        x_e, u_e = None, None
        cost = np.inf
        converged = False
        message = str(e)
        return x_e, u_e, converged, cost, message, {
            "tool": "solve_equilibrium",
            "error": str(e),
            "params": {"strategy": strategy}
        }

    trace_entry = {
        "tool": "solve_equilibrium",
        "params": {
            "initial_guess": initial_guess.tolist(),
            "strategy": strategy
        },
        "result": {
            "x_e": x_e.tolist() if x_e is not None else None,
            "u_e": u_e.tolist() if u_e is not None else None,
            "cost": float(cost),
            "converged": bool(converged)
        }
    }

    return x_e, u_e, converged, cost, message, trace_entry


def compute_jacobian(system_f, x_e, u_e, params, eps=1e-7):
    """
    Tool: Compute linearization matrices A and B via numerical differentiation.

    A = ∂f/∂x|_(x_e,u_e)
    B = ∂f/∂u|_(x_e,u_e)

    Returns:
        A: np.array of shape (n_x, n_x)
        B: np.array of shape (n_x, n_u)
        trace_entry: dict for logging
    """
    if x_e is None or u_e is None:
        trace_entry = {
            "tool": "compute_jacobian",
            "error": "Equilibrium not found",
            "params": {},
            "result": {}
        }
        return None, None, trace_entry

    n_x = len(x_e)
    n_u = len(u_e)
    A = np.zeros((n_x, n_x))
    B = np.zeros((n_x, n_u))

    f0 = system_f(x_e, u_e, params)

    # Compute A: partial derivatives w.r.t. states
    for i in range(n_x):
        x_pert = x_e.copy()
        x_pert[i] += eps
        A[:, i] = (system_f(x_pert, u_e, params) - f0) / eps

    # Compute B: partial derivatives w.r.t. inputs
    for j in range(n_u):
        u_pert = u_e.copy()
        u_pert[j] += eps
        B[:, j] = (system_f(x_e, u_pert, params) - f0) / eps

    trace_entry = {
        "tool": "compute_jacobian",
        "params": {
            "equilibrium": {"x_e": x_e.tolist(), "u_e": u_e.tolist()},
            "eps": eps
        },
        "result": {
            "A": A.tolist(),
            "B": B.tolist(),
            "A_condition_number": float(np.linalg.cond(A))
        }
    }

    return A, B, trace_entry


def analyze_stability(A):
    """
    Tool: Analyze stability via eigenvalue analysis.

    Classification:
    - Asymptotically stable: all Re(λ) < 0
    - Marginally stable: all Re(λ) ≤ 0, at least one Re(λ) = 0
    - Unstable: at least one Re(λ) > 0

    Returns:
        eigenvalues: list of complex eigenvalue strings
        classification: stability classification string
        trace_entry: dict for logging
    """
    if A is None:
        trace_entry = {
            "tool": "analyze_stability",
            "error": "Jacobian not computed",
            "params": {},
            "result": {}
        }
        return None, "unknown", trace_entry

    eigvals_complex = eigvals(A)
    real_parts = np.real(eigvals_complex)
    imag_parts = np.imag(eigvals_complex)

    # Tolerance for numerical zero
    tol = 1e-8

    if np.all(real_parts < -tol):
        classification = "asymptotically stable"
    elif np.all(real_parts < tol) and np.any(np.abs(real_parts) < tol):
        classification = "marginally stable"
    else:
        classification = "unstable"

    # Format eigenvalues for output
    eigenvalues = []
    for ev in eigvals_complex:
        if abs(ev.imag) < tol:
            eigenvalues.append(f"{ev.real:.6f}")
        else:
            eigenvalues.append(f"{ev.real:.6f} + {ev.imag:.6f}j")

    trace_entry = {
        "tool": "analyze_stability",
        "params": {
            "A_shape": list(A.shape)
        },
        "result": {
            "eigenvalues": eigenvalues,
            "real_parts": real_parts.tolist(),
            "imaginary_parts": imag_parts.tolist(),
            "classification": classification
        }
    }

    return eigenvalues, classification, trace_entry


def validate_constraints(x_e, u_e, bounds):
    """
    Tool: Validate physical realizability of equilibrium point.
    
    Checks if equilibrium point lies within specified bounds.
    
    Returns:
        feasible: bool indicating if all constraints are satisfied
        trace_entry: dict for logging
    """
    x_min, x_max = np.array(bounds['x_min']), np.array(bounds['x_max'])
    u_min, u_max = np.array(bounds['u_min']), np.array(bounds['u_max'])
    
    x_feasible = np.all((x_min <= x_e) & (x_e <= x_max))
    u_feasible = np.all((u_min <= u_e) & (u_e <= u_max))
    feasible = bool(x_feasible and u_feasible)
    
    # Detailed constraint violations
    violations = []
    for i, (x_val, x_lo, x_hi) in enumerate(zip(x_e, x_min, x_max)):
        if x_val < x_lo:
            violations.append(f"x[{i}]={x_val:.4f} < {x_lo}")
        elif x_val > x_hi:
            violations.append(f"x[{i}]={x_val:.4f} > {x_hi}")
    
    for j, (u_val, u_lo, u_hi) in enumerate(zip(u_e, u_min, u_max)):
        if u_val < u_lo:
            violations.append(f"u[{j}]={u_val:.4f} < {u_lo}")
        elif u_val > u_hi:
            violations.append(f"u[{j}]={u_val:.4f} > {u_hi}")
    
    trace_entry = {
        "tool": "validate_constraints",
        "params": {
            "equilibrium": {"x_e": x_e.tolist(), "u_e": u_e.tolist()},
            "bounds": bounds
        },
        "result": {
            "feasible": bool(feasible),
            "x_feasible": bool(x_feasible),
            "u_feasible": bool(u_feasible),
            "violations": violations
        }
    }

    return bool(feasible), trace_entry


class ParserAgent:
    """
    Parses system model and extracts state/input variables.
    In production, this would use LLM to parse natural language descriptions.
    """
    def __init__(self):
        self.name = "ParserAgent"
        self.systems = {
            '1': 'msd',
            '2': 'heli',
            '3': 'aircraft'
        }

    def act(self):
        """Interactive system selection."""
        print("\n" + "="*60)
        print("AGENTIC WORKFLOW FOR SYSTEM TRIMMING")
        print("="*60)
        print("\nAvailable Systems:")
        print("1. Mass-Spring-Damper (1-DOF)")
        print("2. 2-DOF Helicopter")
        print("3. Aircraft Longitudinal Dynamics")
        print("4. Other (from customized_system.py)")
        print()

        choice = input("Select system (1-4): ").strip()

        if choice == '4':
            # Load from customized_system.py
            try:
                import customized_system
                available_systems = customized_system.get_available_systems()
                if not available_systems:
                    print("No custom systems available. Defaulting to Mass-Spring-Damper.")
                    choice = '1'
                else:
                    print("\nAvailable Custom Systems:")
                    for i, sys_name in enumerate(available_systems, 1):
                        print(f"{i}. {sys_name}")
                    sub_choice = input(f"Select custom system (1-{len(available_systems)}): ").strip()
                    try:
                        idx = int(sub_choice) - 1
                        if 0 <= idx < len(available_systems):
                            system_name = available_systems[idx]
                            config = customized_system.get_system_config(system_name)
                            # Add system_name to config for compatibility
                            config['system_name'] = system_name
                            return config
                        else:
                            print("Invalid choice. Defaulting to Mass-Spring-Damper.")
                            choice = '1'
                    except ValueError:
                        print("Invalid choice. Defaulting to Mass-Spring-Damper.")
                        choice = '1'
            except ImportError:
                print("Customized systems not available. Defaulting to Mass-Spring-Damper.")
                choice = '1'

        if choice not in self.systems or choice == '4':
            print("Invalid choice. Defaulting to Mass-Spring-Damper.")
            choice = '1'

        system_name = self.systems[choice]

        # Get system configuration
        if system_name == 'msd':
            config = self._configure_msd()
        elif system_name == 'heli':
            config = self._configure_heli()
        elif system_name == 'aircraft':
            config = self._configure_aircraft()
        else:
            raise ValueError(f"Unknown system: {system_name}")

        return config

    def _configure_msd(self):
        """Configure Mass-Spring-Damper system."""
        print("\n--- Mass-Spring-Damper Configuration ---")

        # Default parameters
        params = {
            'm': 1.0,    # kg
            'c': 1.0,    # N·s/m
            'k': 1.0     # N/m
        }

        # Operating conditions
        try:
            desired_force = float(input("Desired equilibrium force [default=1.0]: ") or "1.0")
        except ValueError:
            desired_force = 1.0

        operating_conditions = {'desired_force': desired_force}

        bounds = {
            'x_min': [-10.0, -10.0],
            'x_max': [10.0, 10.0],
            'u_min': [0.0],
            'u_max': [10.0]
        }

        return {
            'system_name': 'msd',
            'system_f': f_msd,
            'params': params,
            'n_x': 2,
            'n_u': 1,
            'operating_conditions': operating_conditions,
            'bounds': bounds,
            'state_vars': ['position [m]', 'velocity [m/s]'],
            'input_vars': ['force [N]'],
            'param_vars': list(params.keys())
        }

    def _configure_heli(self):
        """Configure 2-DOF Helicopter system."""
        print("\n--- 2-DOF Helicopter Configuration ---")

        params = {
            'J_p': 0.0219,
            'D_p': 0.0711,  # Increased damping for faster convergence
            'K_sp': 0.0375,
            'K_pp': 0.0011,
            'K_py': 0.0021,
            'J_y': 0.0220,
            'D_y': 0.220,   # Increased damping for faster convergence
            'K_yp': -0.0027,
            'K_yy': 0.0022,
            'm': 1.0,        # Mass [kg]
            'g': 9.81,       # Gravity [m/s²]
            'l_p': 0.5       # Pitch arm length [m]
        }

        mode = input("Operating mode (1. hover | 2. tilted) [default = 1]: ").strip().lower() or "1"

        if mode == '2':
            try:
                theta_tilt = float(input("Desired pitch tilt [rad, default=0.1]: ") or "0.1")
            except ValueError:
                theta_tilt = 0.1
            operating_conditions = {'mode': 'tilted', 'theta_tilt': theta_tilt}
        else:
            operating_conditions = {'mode': 'hover'}

        bounds = {
            'x_min': [-np.pi/2, -5.0, -10.0, -10.0],
            'x_max': [np.pi/2, 5.0, 10.0, 10.0],
            'u_min': [-24.0, -24.0],
            'u_max': [24.0, 24.0]
        }

        return {
            'system_name': 'heli',
            'system_f': f_heli,
            'params': params,
            'n_x': 4,
            'n_u': 2,
            'operating_conditions': operating_conditions,
            'bounds': bounds,
            'state_vars': ['$\\theta$ [rad]', '$\\psi$ [rad]', '$\\dot{\\theta}$ [rad/s]', '$\\dot{\\psi}$ [rad/s]'],
            'input_vars': ['V_p [V]', 'V_y [V]'],
            'param_vars': list(params.keys())
        }

    def _configure_aircraft(self):
        """Configure Aircraft Longitudinal system."""
        print("\n--- Aircraft Longitudinal Configuration ---")

        params = {
            'm': 1000.0,
            'g': 9.81,
            'rho': 1.225,
            'S': 16.0,
            'c': 1.5,
            'I_yy': 2000.0,
            'T': 2000.0,
            'C_L0': 0.2,
            'C_Lalpha': 5.0,
            'C_Lq': 5.0,
            'C_Ldelta': 1.0,
            'C_D0': 0.03,
            'k': 0.05,
            'C_m0': 0.05,
            'C_malpha': -1.0,
            'C_mq': -10.0,
            'C_mdelta': -1.5
        }

        try:
            airspeed = float(input("Desired airspeed [m/s, default=50.0]: ") or "50.0")
        except ValueError:
            airspeed = 50.0

        operating_conditions = {'airspeed': airspeed}

        bounds = {
            'x_min': [20.0, -0.2, -1.0, -0.2],
            'x_max': [100.0, 0.5, 1.0, 0.5],
            'u_min': [-0.5],
            'u_max': [0.5]
        }

        return {
            'system_name': 'aircraft',
            'system_f': f_aircraft,
            'params': params,
            'n_x': 4,
            'n_u': 1,
            'operating_conditions': operating_conditions,
            'bounds': bounds,
            'state_vars': ['V [m/s]', '$\\alpha$ [rad]', '$\\dot{\\theta}$ [rad/s]', '$\\theta$ [rad]'],
            'input_vars': ['delta_e [rad]'],
            'param_vars': list(params.keys())
        }
