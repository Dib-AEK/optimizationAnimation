import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

class GradientDescentVisualizer:
    def __init__(self, function, gradient, x_range=(-5, 5), y_range=(-5, 5), resolution=100):
        """
        Initialize the gradient descent visualizer.
        
        Args:
            function: Function to minimize (takes x, y and returns scalar)
            gradient: Gradient function (takes x, y and returns (dx, dy))
            x_range: Range for x-axis
            y_range: Range for y-axis
            resolution: Grid resolution for plotting
        """
        self.function = function
        self.gradient = gradient
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        
        # Create meshgrid for plotting
        self.x = np.linspace(x_range[0], x_range[1], resolution)
        self.y = np.linspace(y_range[0], y_range[1], resolution)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Z = self.function(self.X, self.Y)
        
        # Storage for gradient descent path
        self.path_x = []
        self.path_y = []
        self.path_z = []
        
    def gradient_descent(self, start_x, start_y, learning_rate=0.1, max_iterations=100, tolerance=1e-6):
        """
        Perform gradient descent and store the path.
        
        Args:
            start_x, start_y: Starting point
            learning_rate: Step size
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
        """
        x, y = start_x, start_y
        self.path_x = [x]
        self.path_y = [y]
        self.path_z = [self.function(x, y)]
        
        for i in range(max_iterations):
            # Calculate gradient
            grad_x, grad_y = self.gradient(x, y)
            
            # Check for convergence
            if np.sqrt(grad_x**2 + grad_y**2) < tolerance:
                print(f"Converged after {i+1} iterations")
                break
                
            # Update position
            x = x - learning_rate * grad_x
            y = y - learning_rate * grad_y
            
            # Store path
            self.path_x.append(x)
            self.path_y.append(y)
            self.path_z.append(self.function(x, y))
            
        return x, y, self.function(x, y)
    
    def create_animation(self, filename='gradient_descent.mp4', fps=5, interval=500):
        """
        Create an animated visualization of gradient descent.
        
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
        
        # 2D contour plot with heatmap (middle)
        ax2 = fig.add_subplot(132)
        # Create heatmap with more contours
        heatmap = ax2.contourf(self.X, self.Y, self.Z, levels=50, cmap='viridis', alpha=0.8)
        contour = ax2.contour(self.X, self.Y, self.Z, levels=30, colors='black', linewidths=0.5, alpha=0.6)
        ax2.clabel(contour, inline=True, fontsize=6)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Heatmap with Gradient Descent Path')
        ax2.grid(True, alpha=0.3)
        
        # Function value over iterations (right)
        ax3 = fig.add_subplot(133)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Function Value')
        ax3.set_title('Convergence Plot')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Animation control variables
        self.is_paused = False
        self.current_frame = 0
        
        def animate(frame):
            if self.is_paused:
                frame = self.current_frame
            else:
                self.current_frame = frame
                
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
            ax2.set_title('Heatmap with Gradient Descent Path')
            ax2.grid(True, alpha=0.3)
            
            # Plot path up to current frame
            if frame > 0:
                # 3D path
                path_x_current = self.path_x[:frame+1]
                path_y_current = self.path_y[:frame+1]
                path_z_current = self.path_z[:frame+1]
                
                ax1.plot(path_x_current, path_y_current, path_z_current, 
                        'r.-', linewidth=2, markersize=8, label='Gradient Descent Path')
                ax1.scatter(path_x_current[-1], path_y_current[-1], path_z_current[-1], 
                           color='red', s=100, label='Current Position')
                
                # 2D path
                ax2.plot(path_x_current, path_y_current, 'r.-', linewidth=2, markersize=8, 
                        label='Gradient Descent Path')
                ax2.scatter(path_x_current[-1], path_y_current[-1], 
                           color='red', s=100, label='Current Position', zorder=5)
                
                # Convergence plot
                iterations = list(range(len(path_z_current)))
                ax3.plot(iterations, path_z_current, 'b.-', linewidth=2, markersize=6)
                ax3.scatter(iterations[-1], path_z_current[-1], color='red', s=100, zorder=5)
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('Function Value')
                ax3.set_title('Convergence Plot')
                ax3.grid(True, alpha=0.3)
                
                # Add text showing current values
                ax3.text(0.02, 0.98, f'Iteration: {frame}\nf(x,y) = {path_z_current[-1]:.4f}\n'
                        f'x = {path_x_current[-1]:.4f}\ny = {path_y_current[-1]:.4f}',
                        transform=ax3.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
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
                self.current_frame = min(len(self.path_x) - 1, self.current_frame + 1)
                animate(self.current_frame)
                plt.draw()
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(self.path_x), 
                                     interval=interval, repeat=True, blit=False)
        
        # Save animation
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='Gradient Descent Visualizer'), bitrate=1800)
            anim.save(filename, writer=writer)
            print(f"Animation saved as {filename}")
        except Exception as e:
            print(f"Error saving animation: {e}")
        
        print("Controls:")
        print("- Spacebar: Pause/Resume animation")
        print("- Left/Right arrows: Navigate frames when paused")
        
        plt.show()
        
        return anim

# Define test functions and their gradients
def rosenbrock(x, y, a=1, b=100):
    """Rosenbrock function - a classic optimization test function"""
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_gradient(x, y, a=1, b=100):
    """Gradient of Rosenbrock function"""
    dx = -2*(a - x) - 4*b*x*(y - x**2)
    dy = 2*b*(y - x**2)
    return dx, dy

def himmelblau(x, y):
    """Himmelblau's function - has multiple local minima"""
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def himmelblau_gradient(x, y):
    """Gradient of Himmelblau's function"""
    dx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
    dy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
    return dx, dy

def simple_quadratic(x, y):
    """Simple quadratic function"""
    return x**2 + y**2 + 2*x - 4*y + 5

def simple_quadratic_gradient(x, y):
    """Gradient of simple quadratic function"""
    dx = 2*x + 2
    dy = 2*y - 4
    return dx, dy

if __name__ == "__main__":
    print("Gradient Descent Visualization")
    print("=" * 40)
    
    # Choose function to optimize
    print("Available functions:")
    print("1. Simple Quadratic (easy convergence)")
    print("2. Rosenbrock Function (challenging)")
    print("3. Himmelblau's Function (multiple minima)")
    
    choice = input("Choose function (1-3) or press Enter for default (1): ").strip()
    
    if choice == "2":
        func = rosenbrock
        grad_func = rosenbrock_gradient
        x_range = (-2, 2)
        y_range = (-1, 3)
        default_start_x, default_start_y = -1.5, 2.5
        default_learning_rate = 0.001
        title_suffix = "Rosenbrock"
    elif choice == "3":
        func = himmelblau
        grad_func = himmelblau_gradient
        x_range = (-5, 5)
        y_range = (-5, 5)
        default_start_x, default_start_y = -3, -3
        default_learning_rate = 0.01
        title_suffix = "Himmelblau"
    else:  # Default to simple quadratic
        func = simple_quadratic
        grad_func = simple_quadratic_gradient
        x_range = (-5, 5)
        y_range = (-2, 6)
        default_start_x, default_start_y = 3, -1
        default_learning_rate = 0.1
        title_suffix = "Quadratic"
    
    # Get user input for parameters
    print(f"\nDefault starting point: ({default_start_x}, {default_start_y})")
    start_input = input("Enter starting point (x,y) or press Enter for default: ").strip()
    if start_input:
        try:
            start_x, start_y = map(float, start_input.replace('(', '').replace(')', '').split(','))
        except:
            print("Invalid input, using default starting point")
            start_x, start_y = default_start_x, default_start_y
    else:
        start_x, start_y = default_start_x, default_start_y
    
    print(f"Default learning rate: {default_learning_rate}")
    lr_input = input("Enter learning rate or press Enter for default: ").strip()
    if lr_input:
        try:
            learning_rate = float(lr_input)
        except:
            print("Invalid input, using default learning rate")
            learning_rate = default_learning_rate
    else:
        learning_rate = default_learning_rate
    
    # Animation speed control
    print("\nAnimation speed options:")
    print("1. Slow (1 FPS)")
    print("2. Normal (3 FPS)")
    print("3. Fast (5 FPS)")
    speed_choice = input("Choose animation speed (1-3) or press Enter for normal: ").strip()
    
    if speed_choice == "1":
        fps, interval = 1, 1000
    elif speed_choice == "3":
        fps, interval = 5, 200
    else:
        fps, interval = 3, 333
    
    # Create visualizer
    visualizer = GradientDescentVisualizer(func, grad_func, x_range, y_range)
    
    # Run gradient descent
    print(f"\nRunning gradient descent...")
    print(f"Starting point: ({start_x:.2f}, {start_y:.2f})")
    print(f"Initial function value: {func(start_x, start_y):.6f}")
    
    final_x, final_y, final_value = visualizer.gradient_descent(
        start_x, start_y, learning_rate=learning_rate, max_iterations=100
    )
    
    print(f"Final point: ({final_x:.6f}, {final_y:.6f})")
    print(f"Final function value: {final_value:.6f}")
    print(f"Total iterations: {len(visualizer.path_x) - 1}")
    
    # Create animation
    print("\nCreating animation...")
    filename = f'gradient_descent_{title_suffix.lower()}.mp4'
    
    try:
        anim = visualizer.create_animation(filename=filename, fps=fps, interval=interval)
        print(f"Success! Animation saved as '{filename}'")
    except Exception as e:
        print(f"Error creating video: {e}")
        print("Note: Make sure you have ffmpeg installed for video creation.")
        print("You can install it via: conda install ffmpeg or pip install ffmpeg-python")
        
        # Still show the static plot
        plt.show()