# Optimization Animation Visualizer

A Python toolkit for visualizing optimization algorithms through interactive 3D animations. This repository provides implementations of two fundamental optimization methods with real-time visualization capabilities:

1. **Gradient Descent** - Classical first-order optimization algorithm
2. **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)** - Advanced evolutionary optimization algorithm

## What This Repository Does

This project creates animated visualizations of optimization algorithms solving various mathematical functions. Each algorithm generates:

- **3D Surface Plots**: Shows the objective function landscape in three dimensions
- **2D Heatmaps with Contours**: Displays the optimization path overlaid on contour plots
- **Convergence Plots**: Tracks the objective function value over iterations/generations
- **Interactive Controls**: Pause, resume, and step through the optimization process
- **MP4 Export**: Save animations as video files for presentations or documentation

### Supported Test Functions

The visualizer includes several classic optimization benchmark functions:

- **Simple Quadratic**: Easy convergence testing
- **Sphere Function**: Simple convex optimization
- **Rosenbrock Function**: "Banana function" - challenging non-convex optimization
- **Himmelblau's Function**: Multiple local minima
- **Rastrigin Function**: Highly multimodal landscape (CMA-ES only)
- **Ackley Function**: Complex multimodal function (CMA-ES only)

### Key Features

- **Real-time Animation**: Watch optimization algorithms converge step-by-step
- **Interactive Controls**: 
  - Spacebar: Pause/Resume
  - Arrow Keys: Navigate frames when paused
- **Customizable Parameters**: Starting points, learning rates, population sizes
- **Multiple Visualization Angles**: 3D surfaces, 2D projections, convergence plots
- **Export Capabilities**: Save animations as MP4 videos
- **Educational Focus**: Clear visualization of algorithm behavior and convergence patterns

## Requirements

### Python Dependencies

```bash
# Core scientific computing
numpy>=1.21.0
matplotlib>=3.5.0

# CMA-ES implementation
cma>=3.2.0

# 3D plotting (usually included with matplotlib)
# mpl_toolkits.mplot3d
```

### System Dependencies

For video export functionality:
- **FFmpeg**: Required for saving animations as MP4 files


## Usage

### Gradient Descent Visualizer

```bash
python main_gradient_descent.py
```

Interactive prompts will guide you through:
- Function selection (Simple Quadratic, Rosenbrock, Himmelblau)
- Starting point configuration
- Learning rate adjustment
- Animation speed settings

### CMA-ES Visualizer

```bash
python main_cmaes.py
```

Configure:
- Objective function selection (6 different test functions)
- Initial solution guess
- Initial step size (sigma)
- Population size
- Animation parameters

### Example Output

The visualizers generate:
- **Real-time plots**: Three synchronized subplots showing different perspectives
- **Video files**: `gradient_descent_[function].mp4` or `cmaes_optimization_[function].mp4`
- **Console output**: Convergence statistics and final results

### Educational Applications

This tool is ideal for:
- **Machine Learning Courses**: Visualizing optimization landscape navigation
- **Numerical Methods**: Understanding algorithm behavior and convergence
- **Research Presentations**: Creating compelling optimization visualizations
- **Algorithm Comparison**: Side-by-side analysis of different optimization strategies


## License

This project is open source.