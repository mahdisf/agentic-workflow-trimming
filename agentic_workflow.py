import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
import os
from datetime import datetime

# Import agents, tools, and system dynamics
from agents import (
    PlannerAgent, SolverAgent, AnalysisAgent, ParserAgent,
    solve_equilibrium, compute_jacobian, analyze_stability, validate_constraints,
    f_msd, f_heli, f_aircraft
)







# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def workflow(config):
    """
    Main agentic workflow for system trimming.
    
    Process:
    1. Planner determines initial guess and strategy
    2. Solver finds equilibrium iteratively
    3. Analysis performs linearization, stability, and validation
    4. Generate structured output
    """
    trace = []
    
    # Unpack configuration
    system_name = config['system_name']
    system_f = config['system_f']
    params = config['params']
    n_x = config['n_x']
    n_u = config['n_u']
    operating_conditions = config['operating_conditions']
    bounds = config['bounds']
    
    print("\n" + "-"*60)
    print("WORKFLOW EXECUTION")
    print("-"*60)
    
    # Step 1: Planner Agent
    print("\n[1/3] Planner Agent: Determining strategy...")
    planner = PlannerAgent()
    initial_guess, strategy, trace = planner.act(
        operating_conditions, system_name, params, n_x, n_u, bounds, trace
    )
    print(f"  → Initial guess: {initial_guess}")
    print(f"  → Strategy: {strategy}")
    
    # Step 2: Solver Agent
    print("\n[2/3] Solver Agent: Finding equilibrium...")
    solver = SolverAgent(max_retries=10000, tolerance=1e-8)
    x_e, u_e, converged, trace = solver.act(
        system_f, initial_guess, params, n_x, n_u, strategy, bounds, trace
    )
    
    if not converged:
        print("  ✗ Failed to converge!")
        return {
            "error": "Equilibrium solver did not converge",
            "trace": trace,
            "diagnostics": {"converged": False, "feasible": False}
        }
    
    print(f"  ✓ Converged!")
    print(f"  → x_e = {x_e}")
    print(f"  → u_e = {u_e}")
    
    # Step 3: Analysis Agent
    print("\n[3/3] Analysis Agent: Linearization and validation...")
    analysis = AnalysisAgent()
    A, B, eigenvalues, classification, feasible, trace = analysis.act(
        system_f, x_e, u_e, params, bounds, trace
    )
    
    print(f"  → Stability: {classification}")
    print(f"  → Eigenvalues: {eigenvalues}")
    print(f"  → Feasible: {feasible}")
    
    # Step 4: Generate structured output
    n_y = n_x  # Assume full state measurement
    output = {
        "system": {
            "name": system_name,
            "n_states": n_x,
            "n_inputs": n_u,
            "n_outputs": n_y,
            "state_variables": config['state_vars'],
            "input_variables": config['input_vars'],
            "parameters": params
        },
        "equilibrium": {
            "x_e": x_e.tolist() if x_e is not None else None,
            "u_e": u_e.tolist() if u_e is not None else None,
            "y_e": x_e.tolist() if x_e is not None else None  # Assuming y = x (full state output)
        },
        "linearized": {
            "A": A.tolist() if A is not None else None,
            "B": B.tolist() if B is not None else None,
            "C": np.eye(n_x).tolist(),
            "D": np.zeros((n_x, n_u)).tolist()
        },
        "stability": {
            "eigenvalues": eigenvalues,
            "classification": classification
        },
        "trace": trace,
        "diagnostics": {
            "converged": bool(converged),
            "feasible": bool(feasible),
            "timestamp": datetime.now().isoformat()
        }
    }
    
    return output


# ============================================================================
# VISUALIZATION
# ============================================================================

class Plotter:
    """Visualization tool for time-domain simulation."""
    
    def __init__(self, system_f, params, x_e, u_e):
        self.system_f = system_f
        self.params = params
        self.x_e = x_e
        self.u_e = u_e
    
    def simulate_response(self, t_span, x0):
        """Simulate system response from initial condition x0."""
        def ode_func(x, t):
            return self.system_f(x, self.u_e, self.params)
        
        sol = odeint(ode_func, x0, t_span)
        return sol
    
    def plot_time_response(self, t_span, x0, labels, save_path=None):
        """Plot state trajectories over time."""
        sol = self.simulate_response(t_span, x0)
        n = len(labels)

        # Enable mathtext rendering for Greek symbols (no LaTeX required)
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['font.family'] = 'serif'

        fig, axes = plt.subplots(n, 1, figsize=(10, 2.5*n))
        if n == 1:
            axes = [axes]

        for i in range(n):
            axes[i].plot(t_span, sol[:, i], 'b-', linewidth=2, label=f'{labels[i]}')
            axes[i].plot(t_span, [self.x_e[i]] * len(t_span), 'r--',
                        linewidth=1.5, label=f'Equilibrium ({self.x_e[i]:.4f})')
            axes[i].set_xlabel('Time [s]', fontsize=10)
            axes[i].set_ylabel(labels[i], fontsize=10)
            axes[i].legend(loc='best', fontsize=9)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n  → Plot saved to {save_path}")

        plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Step 1: Parse system
    parser = ParserAgent()
    config = parser.act()

    # Step 2: Run workflow
    result = workflow(config)
    
    # Step 3: Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{config['system_name']}_result.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if "error" in result:
        print(f"\n✗ ERROR: {result['error']}")
        print(f"\nResult saved to: {output_file}")
        sys.exit(1)
    
    # Print summary
    print(f"\nSystem: {config['system_name']}")
    print(f"\nEquilibrium Point:")
    for i, (var, val) in enumerate(zip(config['state_vars'], result['equilibrium']['x_e'])):
        print(f"  {var}: {val:.6f}")
    
    print(f"\nEquilibrium Inputs:")
    for i, (var, val) in enumerate(zip(config['input_vars'], result['equilibrium']['u_e'])):
        print(f"  {var}: {val:.6f}")
    
    print(f"\nStability Analysis:")
    print(f"  Classification: {result['stability']['classification']}")
    print(f"  Eigenvalues: {result['stability']['eigenvalues']}")
    
    print(f"\nDiagnostics:")
    print(f"  Converged: {result['diagnostics']['converged']}")
    print(f"  Feasible: {result['diagnostics']['feasible']}")
    
    print(f"\nResult saved to: {output_file}")
    
    # Step 4: Visualization
    print("\n" + "-"*60)
    print("SIMULATION")
    print("-"*60)
    
    plot_choice = input("\nGenerate time response plot? (y/n) [default=y]: ").strip().lower()
    
    if plot_choice != 'n':
        x_e = np.array(result['equilibrium']['x_e'])
        u_e = np.array(result['equilibrium']['u_e'])
        
        plotter = Plotter(config['system_f'], config['params'], x_e, u_e)
        
        # Time span
        t_span = np.linspace(0, 100, 1000)
        
        # Initial condition: perturbed equilibrium
        x0 = x_e + 0.1 * np.random.randn(len(x_e))
        
        plot_file = os.path.join(output_dir, f"{config['system_name']}_response.png")
        plotter.plot_time_response(t_span, x0, config['state_vars'], save_path=plot_file)

    print("\n" + "="*60)
    print("WORKFLOW COMPLETE")
    print("="*60 + "\n")
