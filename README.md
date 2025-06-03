# Multi-Target Tracking with Kalman, EKF, UKF, and Particle Filters

This application is a powerful interactive tool for simulating and analyzing 2D multi-target tracking using Kalman Filter (KF), Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), and Particle Filter (PF). It features real-time visualization, error analysis, and supports both simulated and real sensor data.

## Features
- **Multiple Filters:** Switch between KF, EKF (nonlinear), UKF, and PF.
- **Multi-Target:** Track multiple targets simultaneously.
- **Interactive Visualization:**
  - Real-time 2D plot of true and estimated trajectories
  - Covariance ellipses for uncertainty visualization
  - Full trajectory history
  - Slider to step through simulation history
- **Error Analysis:**
  - Error heatmap: see where the filter makes the largest errors
  - Uncertainty evolution: plot of covariance trace over time
- **Parameter Management:**
  - Save/load simulation parameters (JSON)
  - Load real sensor data (CSV)

## Installation
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd objecttracking
   ```
2. **Install dependencies:**
   ```bash
   pip install numpy matplotlib PyQt5
   ```

## Usage
1. **Run the application:**
   ```bash
   python main.py
   ```
2. **Controls:**
   - Set the number of targets and noise parameters.
   - Choose the filter type (KF, EKF, UKF, PF) from the dropdown.
   - Use Start, Pause, and Reset to control the simulation.
   - Use the slider to step through the simulation history when paused.
   - Save/load parameters with the respective buttons.
   - Load real sensor data (CSV) with the "Load Sensor Data" button.
   - Click "Show Error Heatmap" to visualize where the filter made the largest errors.
   - Click "Show Uncertainty Evolution" to plot the trace of the covariance over time.

## Sensor Data Format
- CSV file, each row = one time step.
- Columns: `x1, y1, x2, y2, ...` (for each target).
- Example for 2 targets:
  ```
  12.3,45.6,78.9,23.4
  13.1,46.2,79.5,24.0
  ...
  ```

## Advanced Analysis Tools
- **Error Heatmap:** Visualizes where the filter made the largest errors on the 2D map.
- **Uncertainty Evolution:** Plots the trace of the covariance matrix over time for each target.
- **Interactive Slider:** Step through the simulation history to analyze any time step.

## Extending the App
- Add new filters, sensor fusion, or advanced data association as needed.
- Export results for further analysis.

## License
MIT License 