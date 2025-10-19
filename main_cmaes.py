import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import cma

class CMAESVisualizer:
    def __init__(self, function, x_range=(-5, 5), y_range=(-5, 5), resolution=100):
        """
        Initialize the CMA-ES visualizer.
        
        Args:
            function: Function to minimize (takes numpy array and returns scalar)
            x_range: Range for x-axis
            y_range: Range for y-axis
            resolution: Grid resolution for plotting
        """
        self.function = function
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        
        # Create meshgrid for plotting
        self.x = np.linspace(x_range[0], x_range[1], resolution)
        self.y = np.linspace(y_range[0], y_range[1], resolution)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Z = np.zeros_like(self.X)
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                self.Z[i, j] = self.function([self.X[i, j], self.Y[i, j]])
        
        # Storage for CMA-ES evolution
        self.generations = []
        self.populations = []
        self.best_solutions = []
        self.best_values = []
        self.sigma_history = []
        self.mean_history = []
        
    def optimize_cmaes(self, initial_guess, initial_sigma=1.0, max_generations=50, population_size=None):
        """
        Perform CMA-ES optimization and store the evolution.
        
        Args:
            initial_guess: Starting point [x, y]
            initial_sigma: Initial step size
            max_generations: Maximum number of generations
            population_size: Population size (None for default)
        """
        # Initialize CMA-ES
        if population_size is None:
            population_size = 4 + int(3 * np.log(len(initial_guess)))
            
        es = cma.CMAEvolutionStrategy(initial_guess, initial_sigma, 
                                     {'maxiter': max_generations, 
                                      'popsize': population_size,
                                      'verbose': -1})  # Reduce verbosity
        
        generation = 0
        while not es.stop():
            # Generate new population
            solutions = es.ask()
            
            # Evaluate fitness
            fitness_values = [self.function(x) for x in solutions]
            
            # Update evolution strategy
            es.tell(solutions, fitness_values)
            
            # Store data for visualization
            self.generations.append(generation)
            self.populations.append(np.array(solutions))
            self.best_solutions.append(es.result[0].copy())
            self.best_values.append(es.result[1])
            self.sigma_history.append(es.sigma)
            self.mean_history.append(es.mean.copy())
            
            generation += 1
            
        print(f"CMA-ES completed after {generation} generations")
        print(f"Best solution: {es.result[0]}")
        print(f"Best value: {es.result[1]:.6f}")
        
        return es.result
    
    def create_animation(self, filename='cmaes_optimization.mp4', fps=2, interval=1000):
        """
        Create an animated visualization of CMA-ES optimization.
        
        Args:
            filename: Output video filename
            fps: Frames per second
            interval: Interval between frames in milliseconds
        """
        fig = plt.figure(figsize=(18, 6))
        
        # 3D surface plot (left)
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_surface(self.X, self.Y, self.Z, alpha=0.6, cmap='viridis')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('f(X,Y)')
        ax1.set_title('3D Function Surface')
        
        # 2D heatmap with population (middle)
        ax2 = fig.add_subplot(132)
        # Create heatmap with more contours
        heatmap = ax2.contourf(self.X, self.Y, self.Z, levels=50, cmap='viridis', alpha=0.8)
        contour = ax2.contour(self.X, self.Y, self.Z, levels=30, colors='black', linewidths=0.5, alpha=0.6)
        ax2.clabel(contour, inline=True, fontsize=6)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Population Evolution')
        ax2.grid(True, alpha=0.3)
        
        # Convergence plot (right)
        ax3 = fig.add_subplot(133)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Best Function Value')
        ax3.set_title('Convergence Plot')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        plt.tight_layout()
        
        # Animation control variables
        self.is_paused = False
        self.current_frame = 0
        
        def animate(frame):
            if self.is_paused:
                frame = self.current_frame
            else:
                self.current_frame = frame
                
            if frame >= len(self.populations):
                return
                
            # Clear previous plots
            ax1.clear()
            ax2.clear()
            ax3.clear()
            
            # Redraw 3D surface
            ax1.plot_surface(self.X, self.Y, self.Z, alpha=0.6, cmap='viridis')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('f(X,Y)')
            ax1.set_title('3D Function Surface')
            
            # Redraw heatmap with more contours
            heatmap = ax2.contourf(self.X, self.Y, self.Z, levels=50, cmap='viridis', alpha=0.8)
            contour = ax2.contour(self.X, self.Y, self.Z, levels=30, colors='black', linewidths=0.5, alpha=0.6)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title('Population Evolution')
            ax2.grid(True, alpha=0.3)
            
            # Plot current population
            current_pop = self.populations[frame]
            ax2.scatter(current_pop[:, 0], current_pop[:, 1], 
                       c='red', s=30, alpha=0.7, label='Population')
            
            # Plot best solution path up to current frame
            if frame > 0:
                best_x = [sol[0] for sol in self.best_solutions[:frame+1]]
                best_y = [sol[1] for sol in self.best_solutions[:frame+1]]
                ax2.plot(best_x, best_y, 'b-', linewidth=2, alpha=0.8, label='Best Solution Path')
            
            # Plot current best solution
            current_best = self.best_solutions[frame]
            ax2.scatter(current_best[0], current_best[1], 
                       c='blue', s=100, marker='*', label='Best Solution', zorder=5)
            
            # Plot mean
            current_mean = self.mean_history[frame]
            ax2.scatter(current_mean[0], current_mean[1], 
                       c='green', s=80, marker='x', linewidth=3, label='Mean', zorder=5)
            
            ax2.legend(loc='upper right')
            ax2.set_xlim(self.x_range)
            ax2.set_ylim(self.y_range)
            
            # Plot convergence
            generations_so_far = self.generations[:frame+1]
            best_values_so_far = self.best_values[:frame+1]
            ax3.plot(generations_so_far, best_values_so_far, 'b.-', linewidth=2, markersize=6)
            ax3.scatter(generations_so_far[-1], best_values_so_far[-1], 
                       color='red', s=100, zorder=5)
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Best Function Value')
            ax3.set_title('Convergence Plot')
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
            
            # Add information text
            info_text = (f'Generation: {frame}\n'
                        f'Best Value: {self.best_values[frame]:.6f}\n'
                        f'Best X: {current_best[0]:.4f}\n'
                        f'Best Y: {current_best[1]:.4f}\n'
                        f'σ: {self.sigma_history[frame]:.4f}\n'
                        f'Population Size: {len(current_pop)}')
            
            ax3.text(0.02, 0.98, info_text, transform=ax3.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8))
            
            # Update 3D plot with current best
            if frame > 0:
                best_x_3d = [sol[0] for sol in self.best_solutions[:frame+1]]
                best_y_3d = [sol[1] for sol in self.best_solutions[:frame+1]]
                best_z_3d = [val for val in self.best_values[:frame+1]]
                ax1.plot(best_x_3d, best_y_3d, best_z_3d, 'r.-', linewidth=2, markersize=8)
                ax1.scatter(current_best[0], current_best[1], self.best_values[frame], 
                           color='red', s=100)
            
            plt.tight_layout()
        
        # Animation control functions
        def on_key(event):
            if event.key == ' ':  # Spacebar to pause/unpause
                self.is_paused = not self.is_paused
                print("Animation paused" if self.is_paused else "Animation resumed")
            elif event.key == 'left' and self.is_paused:
                self.current_frame = max(0, self.current_frame - 1)
                animate(self.current_frame)
                plt.draw()
            elif event.key == 'right' and self.is_paused:
                self.current_frame = min(len(self.populations) - 1, self.current_frame + 1)
                animate(self.current_frame)
                plt.draw()
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(self.populations), 
                                     interval=interval, repeat=True, blit=False)
        
        # Save animation
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='CMA-ES Visualizer'), bitrate=1800)
            anim.save(filename, writer=writer)
            print(f"Animation saved as {filename}")
        except Exception as e:
            print(f"Error saving animation: {e}")
        
        print("Controls:")
        print("- Spacebar: Pause/Resume animation")
        print("- Left/Right arrows: Navigate frames when paused")
        
        plt.show()
        return anim

# Define test functions (modified to work with numpy arrays)
def rosenbrock_cma(x, a=1, b=100):
    """Rosenbrock function for CMA-ES (takes numpy array)"""
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def himmelblau_cma(x):
    """Himmelblau's function for CMA-ES (takes numpy array)"""
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def simple_quadratic_cma(x):
    """Simple quadratic function for CMA-ES (takes numpy array)"""
    return x[0]**2 + x[1]**2 + 2*x[0] - 4*x[1] + 5

def rastrigin_cma(x, A=10):
    """Rastrigin function - multimodal test function"""
    n = len(x)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def ackley_cma(x, a=20, b=0.2, c=2*np.pi):
    """Ackley function - another multimodal test function"""
    n = len(x)
    return (-a * np.exp(-b * np.sqrt(sum([xi**2 for xi in x]) / n)) -
            np.exp(sum([np.cos(c * xi) for xi in x]) / n) + a + np.exp(1))

def sphere_cma(x):
    """Simple sphere function"""
    return sum([xi**2 for xi in x])

if __name__ == "__main__":
    print("CMA-ES Optimization Visualization")
    print("=" * 50)
    
    # Choose function to optimize
    print("Available functions:")
    print("1. Simple Quadratic (easy convergence)")
    print("2. Sphere Function (simple)")
    print("3. Rosenbrock Function (challenging)")
    print("4. Himmelblau's Function (multiple minima)")
    print("5. Rastrigin Function (highly multimodal)")
    print("6. Ackley Function (multimodal)")
    
    choice = input("Choose function (1-6) or press Enter for default (1): ").strip()
    
    if choice == "2":
        func = sphere_cma
        x_range = (-5, 5)
        y_range = (-5, 5)
        default_initial_guess = [3, -2]
        default_initial_sigma = 2.0
        title_suffix = "Sphere"
    elif choice == "3":
        func = rosenbrock_cma
        x_range = (-2, 2)
        y_range = (-1, 3)
        default_initial_guess = [-1.5, 2.5]
        default_initial_sigma = 1.0
        title_suffix = "Rosenbrock"
    elif choice == "4":
        func = himmelblau_cma
        x_range = (-5, 5)
        y_range = (-5, 5)
        default_initial_guess = [-3, -3]
        default_initial_sigma = 2.0
        title_suffix = "Himmelblau"
    elif choice == "5":
        func = rastrigin_cma
        x_range = (-5.12, 5.12)
        y_range = (-5.12, 5.12)
        default_initial_guess = [3, -2]
        default_initial_sigma = 2.0
        title_suffix = "Rastrigin"
    elif choice == "6":
        func = ackley_cma
        x_range = (-5, 5)
        y_range = (-5, 5)
        default_initial_guess = [3, -2]
        default_initial_sigma = 2.0
        title_suffix = "Ackley"
    else:  # Default to simple quadratic
        func = simple_quadratic_cma
        x_range = (-5, 5)
        y_range = (-2, 6)
        default_initial_guess = [3, -1]
        default_initial_sigma = 1.5
        title_suffix = "Quadratic"
    
    # Get user input for parameters
    print(f"\nDefault starting point: {default_initial_guess}")
    start_input = input("Enter starting point [x,y] or press Enter for default: ").strip()
    if start_input:
        try:
            # Handle different input formats
            start_input = start_input.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
            initial_guess = list(map(float, start_input.split(',')))
            if len(initial_guess) != 2:
                raise ValueError("Must provide exactly 2 values")
        except:
            print("Invalid input, using default starting point")
            initial_guess = default_initial_guess
    else:
        initial_guess = default_initial_guess
    
    print(f"Default initial sigma: {default_initial_sigma}")
    sigma_input = input("Enter initial sigma or press Enter for default: ").strip()
    if sigma_input:
        try:
            initial_sigma = float(sigma_input)
        except:
            print("Invalid input, using default sigma")
            initial_sigma = default_initial_sigma
    else:
        initial_sigma = default_initial_sigma
    
    print("Default population size: Auto (4 + 3*ln(dimensions))")
    pop_input = input("Enter population size or press Enter for auto: ").strip()
    if pop_input:
        try:
            population_size = int(pop_input)
        except:
            print("Invalid input, using automatic population size")
            population_size = None
    else:
        population_size = None
    
    # Animation speed control
    print("\nAnimation speed options:")
    print("1. Slow (1 FPS)")
    print("2. Normal (2 FPS)")
    print("3. Fast (4 FPS)")
    speed_choice = input("Choose animation speed (1-3) or press Enter for normal: ").strip()
    
    if speed_choice == "1":
        fps, interval = 1, 1000
    elif speed_choice == "3":
        fps, interval = 4, 250
    else:
        fps, interval = 2, 500
    
    # Create visualizer
    visualizer = CMAESVisualizer(func, x_range, y_range)
    
    # Run CMA-ES optimization
    print(f"\nRunning CMA-ES optimization...")
    print(f"Starting point: {initial_guess}")
    print(f"Initial function value: {func(initial_guess):.6f}")
    print(f"Initial sigma: {initial_sigma}")
    
    result = visualizer.optimize_cmaes(
        initial_guess, initial_sigma=initial_sigma, 
        max_generations=30, population_size=population_size
    )
    
    print(f"Final point: [{result[0][0]:.6f}, {result[0][1]:.6f}]")
    print(f"Final function value: {result[1]:.6f}")
    print(f"Total generations: {len(visualizer.generations)}")
    
    # Create animation
    print("\nCreating animation...")
    filename = f'cmaes_optimization_{title_suffix.lower()}.mp4'
    
    try:
        anim = visualizer.create_animation(filename=filename, fps=fps, interval=interval)
        print(f"Success! Animation saved as '{filename}'")
    except Exception as e:
        print(f"Error creating video: {e}")
        print("Note: Make sure you have ffmpeg installed for video creation.")
        
        # Still show the static plot
        plt.show()