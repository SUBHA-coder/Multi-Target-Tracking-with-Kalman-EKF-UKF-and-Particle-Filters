import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QSpinBox, QDoubleSpinBox, QTextEdit, QGroupBox, QFileDialog, QCheckBox, QComboBox, QSlider
)
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import json
from matplotlib.patches import Ellipse
from matplotlib import cm
from matplotlib.colors import Normalize

# --- Kalman Filter Class ---
class KalmanFilter2D:
    def __init__(self, dt=1.0, process_noise=1e-3, measurement_noise=1e-1):
        # State vector: [x, y, vx, vy]
        self.dt = dt

        # State transition matrix (constant velocity model)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Measurement matrix (we measure positions x, y only)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Covariance matrices
        self.Q = process_noise * np.eye(4)  # Process noise covariance
        self.R = measurement_noise * np.eye(2)  # Measurement noise covariance

        self.x = np.zeros((4, 1))  # Initial state
        self.P = np.eye(4)         # Initial covariance

    def predict(self):
        # Predict state and covariance forward
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # z: measurement vector shape (2,1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def set_state(self, x_init):
        self.x = x_init.reshape(4,1)

    def set_process_noise(self, q):
        self.Q = q * np.eye(4)

    def set_measurement_noise(self, r):
        self.R = r * np.eye(2)

# --- Extended Kalman Filter Class ---
class ExtendedKalmanFilter2D:
    def __init__(self, dt=1.0, process_noise=1e-3, measurement_noise=1e-1):
        self.dt = dt
        self.Q = process_noise * np.eye(4)
        self.R = measurement_noise * np.eye(2)
        self.x = np.zeros((4, 1))
        self.P = np.eye(4)

    def f(self, x):
        # Nonlinear motion: simple coordinated turn (for demo)
        # x = [x, y, v, theta]
        x_new = np.zeros_like(x)
        v = x[2,0]
        theta = x[3,0]
        x_new[0,0] = x[0,0] + v * np.cos(theta) * self.dt
        x_new[1,0] = x[1,0] + v * np.sin(theta) * self.dt
        x_new[2,0] = v
        x_new[3,0] = theta
        return x_new

    def F_jacobian(self, x):
        v = x[2,0]
        theta = x[3,0]
        F = np.eye(4)
        F[0,2] = np.cos(theta) * self.dt
        F[0,3] = -v * np.sin(theta) * self.dt
        F[1,2] = np.sin(theta) * self.dt
        F[1,3] = v * np.cos(theta) * self.dt
        return F

    def h(self, x):
        # Measurement: only position
        return x[0:2]

    def H_jacobian(self, x):
        H = np.zeros((2,4))
        H[0,0] = 1
        H[1,1] = 1
        return H

    def predict(self):
        self.x = self.f(self.x)
        F = self.F_jacobian(self.x)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        H = self.H_jacobian(self.x)
        y = z - self.h(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def set_state(self, x_init):
        self.x = x_init.reshape(4,1)

    def set_process_noise(self, q):
        self.Q = q * np.eye(4)

    def set_measurement_noise(self, r):
        self.R = r * np.eye(2)

# --- Unscented Kalman Filter Class (basic, for demo) ---
class UnscentedKalmanFilter2D:
    def __init__(self, dt=1.0, process_noise=1e-3, measurement_noise=1e-1):
        self.dt = dt
        self.Q = process_noise * np.eye(4)
        self.R = measurement_noise * np.eye(2)
        self.x = np.zeros((4, 1))
        self.P = np.eye(4)
    def predict(self):
        # For demo, use same as linear KF
        F = np.array([[1, 0, self.dt, 0],
                      [0, 1, 0, self.dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
    def update(self, z):
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
    def set_state(self, x_init):
        self.x = x_init.reshape(4,1)
    def set_process_noise(self, q):
        self.Q = q * np.eye(4)
    def set_measurement_noise(self, r):
        self.R = r * np.eye(2)

# --- Particle Filter Class (basic, for demo) ---
class ParticleFilter2D:
    def __init__(self, dt=1.0, process_noise=1e-3, measurement_noise=1e-1, num_particles=100):
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.num_particles = num_particles
        self.particles = np.zeros((num_particles, 4))
        self.weights = np.ones(num_particles) / num_particles
    def set_state(self, x_init):
        self.particles = np.tile(x_init, (self.num_particles, 1))
        self.weights = np.ones(self.num_particles) / self.num_particles
    def set_process_noise(self, q):
        self.process_noise = q
    def set_measurement_noise(self, r):
        self.measurement_noise = r
    def predict(self):
        noise = np.random.normal(0, np.sqrt(self.process_noise), (self.num_particles, 4))
        self.particles[:,0] += self.particles[:,2] * self.dt + noise[:,0]
        self.particles[:,1] += self.particles[:,3] * self.dt + noise[:,1]
        self.particles[:,2] += noise[:,2]
        self.particles[:,3] += noise[:,3]
    def update(self, z):
        # z shape: (2,1)
        dists = np.linalg.norm(self.particles[:,0:2] - z.T, axis=1)
        self.weights = np.exp(-0.5 * (dists**2) / self.measurement_noise)
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)
        # Resample
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
    @property
    def x(self):
        # Return mean state as (4,1)
        return np.mean(self.particles, axis=0).reshape(4,1)
    @property
    def P(self):
        # Return covariance of position as (4,4)
        return np.cov(self.particles.T)

# --- Main Window ---
class KalmanTrackerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Kalman Filter Multi-Target Tracker")
        self.setGeometry(100, 100, 1000, 700)

        self.num_targets = 3
        self.dt = 1.0
        self.process_noise = 1e-3
        self.measurement_noise = 1e-1
        self.is_running = False
        self.targets = []
        self.filters = []
        self.trajectories = []  # True target history
        self.estimate_trajectories = []  # Filter estimate history
        self.filter_type = 'KF'  # 'KF' or 'EKF'
        self.sensor_data = None
        self.sensor_data_idx = 0
        self.history = []  # Store (trajectories, estimates, covariances) for each step
        self.slider_enabled = False
        self.error_map = None  # 2D histogram for error heatmap
        self.error_map_bins = (50, 50)
        self.error_map_range = [[0, 100], [0, 100]]
        self.uncertainty_history = [[] for _ in range(self.num_targets)]

        self.initUI()

        # Initialize timer BEFORE reset_simulation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.setInterval(1000)  # Update every 1 second

        self.reset_simulation()

    def initUI(self):
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Left side: controls
        controls_layout = QVBoxLayout()

        # Number of targets
        nt_group = QGroupBox("Number of Targets")
        nt_layout = QHBoxLayout()
        self.spin_num_targets = QSpinBox()
        self.spin_num_targets.setRange(1, 10)
        self.spin_num_targets.setValue(self.num_targets)
        self.spin_num_targets.valueChanged.connect(self.change_num_targets)
        nt_layout.addWidget(QLabel("Targets:"))
        nt_layout.addWidget(self.spin_num_targets)
        nt_group.setLayout(nt_layout)
        controls_layout.addWidget(nt_group)

        # Filter type toggle
        filter_group = QGroupBox("Filter Type")
        filter_layout = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Kalman Filter", "Extended Kalman Filter", "Unscented Kalman Filter", "Particle Filter"])
        self.filter_combo.currentIndexChanged.connect(self.change_filter_type)
        filter_layout.addWidget(self.filter_combo)
        filter_group.setLayout(filter_layout)
        controls_layout.addWidget(filter_group)

        # Noise parameters
        noise_group = QGroupBox("Noise Parameters")
        noise_layout = QVBoxLayout()

        proc_noise_layout = QHBoxLayout()
        proc_noise_layout.addWidget(QLabel("Process Noise (Q):"))
        self.proc_noise_spin = QDoubleSpinBox()
        self.proc_noise_spin.setDecimals(6)
        self.proc_noise_spin.setRange(1e-6, 0.1)
        self.proc_noise_spin.setSingleStep(1e-4)
        self.proc_noise_spin.setValue(self.process_noise)
        self.proc_noise_spin.valueChanged.connect(self.change_process_noise)
        proc_noise_layout.addWidget(self.proc_noise_spin)
        noise_layout.addLayout(proc_noise_layout)

        meas_noise_layout = QHBoxLayout()
        meas_noise_layout.addWidget(QLabel("Measurement Noise (R):"))
        self.meas_noise_spin = QDoubleSpinBox()
        self.meas_noise_spin.setDecimals(6)
        self.meas_noise_spin.setRange(1e-6, 0.1)
        self.meas_noise_spin.setSingleStep(1e-4)
        self.meas_noise_spin.setValue(self.measurement_noise)
        self.meas_noise_spin.valueChanged.connect(self.change_measurement_noise)
        meas_noise_layout.addWidget(self.meas_noise_spin)
        noise_layout.addLayout(meas_noise_layout)

        noise_group.setLayout(noise_layout)
        controls_layout.addWidget(noise_group)

        # Start / Pause / Reset buttons
        buttons_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.start_simulation)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self.pause_simulation)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_simulation)
        buttons_layout.addWidget(self.btn_start)
        buttons_layout.addWidget(self.btn_pause)
        buttons_layout.addWidget(self.btn_reset)
        controls_layout.addLayout(buttons_layout)

        # Save/Load buttons
        save_load_layout = QHBoxLayout()
        self.btn_save = QPushButton("Save Params")
        self.btn_save.clicked.connect(self.save_params)
        self.btn_load = QPushButton("Load Params")
        self.btn_load.clicked.connect(self.load_params)
        # Real data button
        self.btn_load_data = QPushButton("Load Sensor Data")
        self.btn_load_data.clicked.connect(self.load_sensor_data)
        save_load_layout.addWidget(self.btn_save)
        save_load_layout.addWidget(self.btn_load)
        save_load_layout.addWidget(self.btn_load_data)
        controls_layout.addLayout(save_load_layout)

        # Logs/Stats text box
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        controls_layout.addWidget(QLabel("Filter Logs and Statistics"))
        controls_layout.addWidget(self.log_text)

        # Create figure and canvas BEFORE using self.canvas
        self.fig = Figure(figsize=(7, 7))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_title("2D Target Tracking")
        self.ax.set_xlabel("X position")
        self.ax.set_ylabel("Y position")
        self.ax.grid(True)
        # Add slider for interactive time stepping
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.slider_step)
        # Add button for error heatmap
        self.btn_heatmap = QPushButton("Show Error Heatmap")
        self.btn_heatmap.clicked.connect(self.show_error_heatmap)
        controls_layout.addWidget(self.btn_heatmap)
        # Add button for uncertainty evolution
        self.btn_uncertainty = QPushButton("Show Uncertainty Evolution")
        self.btn_uncertainty.clicked.connect(self.show_uncertainty_evolution)
        controls_layout.addWidget(self.btn_uncertainty)
        # Add slider below the plot
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.canvas)
        plot_layout.addWidget(self.slider)
        layout.addLayout(controls_layout, 2)
        layout.addLayout(plot_layout, 5)

    def reset_simulation(self):
        self.is_running = False
        self.timer.stop()
        self.targets = []
        self.filters = []
        self.trajectories = [[] for _ in range(self.num_targets)]
        self.estimate_trajectories = [[] for _ in range(self.num_targets)]
        self.covariances = [[] for _ in range(self.num_targets)]
        self.time_step = 0
        self.log_text.clear()
        self.history = []
        self.error_map = np.zeros(self.error_map_bins)
        self.uncertainty_history = [[] for _ in range(self.num_targets)]
        for i in range(self.num_targets):
            if self.sensor_data is not None and self.sensor_data.shape[1] >= 2*self.num_targets:
                # Use real data for initial state
                state = np.zeros(4)
                state[0] = self.sensor_data[0, 2*i]
                state[1] = self.sensor_data[0, 2*i+1]
                state[2] = 0
                state[3] = 0
            elif self.filter_type == 'EKF':
                x = np.random.uniform(10, 90)
                y = np.random.uniform(10, 90)
                v = np.random.uniform(1, 3)
                theta = np.random.uniform(0, 2*np.pi)
                state = np.array([x, y, v, theta])
            else:
                state = np.array([
                    np.random.uniform(10, 90),
                    np.random.uniform(10, 90),
                    np.random.uniform(-2, 2),
                    np.random.uniform(-2, 2),
                ])
            self.targets.append(state)
            if self.filter_type == 'EKF':
                kf = ExtendedKalmanFilter2D(
                    dt=self.dt,
                    process_noise=self.process_noise,
                    measurement_noise=self.measurement_noise,
                )
            elif self.filter_type == 'UKF':
                kf = UnscentedKalmanFilter2D(
                    dt=self.dt,
                    process_noise=self.process_noise,
                    measurement_noise=self.measurement_noise,
                )
            elif self.filter_type == 'PF':
                kf = ParticleFilter2D(
                    dt=self.dt,
                    process_noise=self.process_noise,
                    measurement_noise=self.measurement_noise,
                )
            else:
                kf = KalmanFilter2D(
                    dt=self.dt,
                    process_noise=self.process_noise,
                    measurement_noise=self.measurement_noise,
                )
            kf.set_state(state)
            self.filters.append(kf)
            self.trajectories[i].append(state[0:2].copy())
            self.estimate_trajectories[i].append(state[0:2].copy())
            if hasattr(kf, 'P'):
                self.covariances[i].append(kf.P[0:2,0:2].copy())
            else:
                self.covariances[i].append(np.eye(2))
        self.history.append(([[p.copy() for p in traj] for traj in self.trajectories],
                             [[e.copy() for e in est] for est in self.estimate_trajectories],
                             [[c.copy() for c in cov] for cov in self.covariances]))
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.plot_state()

    def start_simulation(self):
        if not self.is_running:
            self.is_running = True
            self.timer.start()

    def pause_simulation(self):
        self.is_running = False
        self.timer.stop()

    def change_num_targets(self, val):
        self.num_targets = val
        self.reset_simulation()

    def change_process_noise(self, val):
        self.process_noise = val
        for kf in self.filters:
            kf.set_process_noise(val)

    def change_measurement_noise(self, val):
        self.measurement_noise = val
        for kf in self.filters:
            kf.set_measurement_noise(val)

    def change_filter_type(self, idx):
        if idx == 0:
            self.filter_type = 'KF'
        elif idx == 1:
            self.filter_type = 'EKF'
        elif idx == 2:
            self.filter_type = 'UKF'
        elif idx == 3:
            self.filter_type = 'PF'
        self.reset_simulation()

    def save_params(self):
        params = {
            'num_targets': self.num_targets,
            'process_noise': self.process_noise,
            'measurement_noise': self.measurement_noise,
            'filter_type': self.filter_type,
        }
        fname, _ = QFileDialog.getSaveFileName(self, "Save Parameters", "params.json", "JSON Files (*.json)")
        if fname:
            with open(fname, 'w') as f:
                json.dump(params, f, indent=2)

    def load_params(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Parameters", "", "JSON Files (*.json)")
        if fname:
            with open(fname, 'r') as f:
                params = json.load(f)
            self.spin_num_targets.setValue(params.get('num_targets', self.num_targets))
            self.proc_noise_spin.setValue(params.get('process_noise', self.process_noise))
            self.meas_noise_spin.setValue(params.get('measurement_noise', self.measurement_noise))
            filter_map = {'KF': 0, 'EKF': 1, 'UKF': 2, 'PF': 3}
            idx = filter_map.get(params.get('filter_type', 'KF'), 0)
            self.filter_combo.setCurrentIndex(idx)

    def load_sensor_data(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Sensor Data", "", "CSV Files (*.csv)")
        if fname:
            self.sensor_data = np.loadtxt(fname, delimiter=',')
            self.sensor_data_idx = 0
            # Adjust number of targets to match data
            n_targets_in_data = self.sensor_data.shape[1] // 2
            self.spin_num_targets.setValue(n_targets_in_data)
            self.reset_simulation()

    def plot_covariance_ellipse(self, mean, cov, ax, n_std=2.0, **kwargs):
        # Only use the 2x2 position covariance
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        width, height = 2 * n_std * np.sqrt(vals)
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)
        ax.add_patch(ellipse)

    def update_simulation(self):
        self.time_step += 1
        self.ax.clear()
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_title(f"2D Target Tracking - Step {self.time_step}")
        self.ax.set_xlabel("X position")
        self.ax.set_ylabel("Y position")
        self.ax.grid(True)
        total_error = 0
        n_targets = self.num_targets
        if self.sensor_data is not None:
            n_targets = min(self.num_targets, self.sensor_data.shape[1] // 2)
        for idx in range(n_targets):
            if self.sensor_data is not None and self.sensor_data_idx < self.sensor_data.shape[0]:
                # Use real data for measurement
                x = self.sensor_data[self.sensor_data_idx, 2*idx]
                y = self.sensor_data[self.sensor_data_idx, 2*idx+1]
                self.targets[idx][0:2] = [x, y]
                z = np.array([[x], [y]])
                kf = self.filters[idx]
                kf.predict()
                kf.update(z)
                est_x, est_y = kf.x[0,0], kf.x[1,0]
            elif self.filter_type == 'EKF':
                x, y, v, theta = self.targets[idx]
                x += v * np.cos(theta) * self.dt
                y += v * np.sin(theta) * self.dt
                theta += np.random.normal(0, 0.05)
                if x < 0 or x > 100:
                    v = -v
                    x = np.clip(x, 0, 100)
                if y < 0 or y > 100:
                    v = -v
                    y = np.clip(y, 0, 100)
                self.targets[idx] = np.array([x, y, v, theta])
                meas_noise = np.random.multivariate_normal(mean=[0,0], cov=self.measurement_noise*np.eye(2))
                z = np.array([[x + meas_noise[0]], [y + meas_noise[1]]])
                kf = self.filters[idx]
                kf.predict()
                kf.update(z)
                est_x, est_y = kf.x[0,0], kf.x[1,0]
            elif self.filter_type == 'UKF':
                state = self.targets[idx]
                x, y, vx, vy = state
                x += vx * self.dt
                y += vy * self.dt
                if x < 0 or x > 100:
                    vx = -vx
                if y < 0 or y > 100:
                    vy = -vy
                self.targets[idx] = np.array([x, y, vx, vy])
                meas_noise = np.random.multivariate_normal(mean=[0,0], cov=self.measurement_noise*np.eye(2))
                z = np.array([[x + meas_noise[0]], [y + meas_noise[1]]])
                kf = self.filters[idx]
                kf.predict()
                kf.update(z)
                est_x, est_y = kf.x[0,0], kf.x[1,0]
            elif self.filter_type == 'PF':
                state = self.targets[idx]
                x, y, vx, vy = state
                x += vx * self.dt
                y += vy * self.dt
                if x < 0 or x > 100:
                    vx = -vx
                if y < 0 or y > 100:
                    vy = -vy
                self.targets[idx] = np.array([x, y, vx, vy])
                meas_noise = np.random.multivariate_normal(mean=[0,0], cov=self.measurement_noise*np.eye(2))
                z = np.array([[x + meas_noise[0]], [y + meas_noise[1]]])
                kf = self.filters[idx]
                kf.predict()
                kf.update(z)
                est_x, est_y = kf.x[0,0], kf.x[1,0]
            else:
                state = self.targets[idx]
                x, y, vx, vy = state
                x += vx * self.dt
                y += vy * self.dt
                if x < 0 or x > 100:
                    vx = -vx
                if y < 0 or y > 100:
                    vy = -vy
                self.targets[idx] = np.array([x, y, vx, vy])
                meas_noise = np.random.multivariate_normal(mean=[0,0], cov=self.measurement_noise*np.eye(2))
                z = np.array([[x + meas_noise[0]], [y + meas_noise[1]]])
                kf = self.filters[idx]
                kf.predict()
                kf.update(z)
                est_x, est_y = kf.x[0,0], kf.x[1,0]
            self.trajectories[idx].append(self.targets[idx][0:2].copy())
            self.estimate_trajectories[idx].append(np.array([est_x, est_y]))
            traj = np.array(self.trajectories[idx])
            self.ax.plot(traj[:,0], traj[:,1], 'g-', alpha=0.5)
            etraj = np.array(self.estimate_trajectories[idx])
            self.ax.plot(etraj[:,0], etraj[:,1], 'b--', alpha=0.5)
            self.ax.plot(traj[-1,0], traj[-1,1], 'go', label=f"True Target {idx+1}" if idx == 0 else "")
            self.ax.plot(etraj[-1,0], etraj[-1,1], 'bx', label=f"Estimate {idx+1}" if idx == 0 else "")
            error = np.linalg.norm(np.array([est_x, est_y]) - self.targets[idx][0:2])
            total_error += error
            # Add error to heatmap
            x_pos, y_pos = self.targets[idx][0:2]
            x_bin = int((x_pos - self.error_map_range[0][0]) / (self.error_map_range[0][1] - self.error_map_range[0][0]) * (self.error_map_bins[0] - 1))
            y_bin = int((y_pos - self.error_map_range[1][0]) / (self.error_map_range[1][1] - self.error_map_range[1][0]) * (self.error_map_bins[1] - 1))
            if 0 <= x_bin < self.error_map_bins[0] and 0 <= y_bin < self.error_map_bins[1]:
                self.error_map[x_bin, y_bin] += error
            # Store uncertainty (trace of covariance)
            if self.filter_type in ['KF', 'EKF', 'UKF']:
                kf = self.filters[idx]
                cov = kf.P[0:2,0:2]
                self.uncertainty_history[idx].append(np.trace(cov))
            else:
                self.uncertainty_history[idx].append(0)
        self.ax.legend(loc="upper right")
        self.canvas.draw()
        avg_error = total_error / n_targets if n_targets > 0 else 0
        self.log_text.append(f"Step {self.time_step}: Avg Estimation Error = {avg_error:.3f}")
        if self.sensor_data is not None:
            self.sensor_data_idx += 1
        # Store history for slider
        self.history.append(([[p.copy() for p in traj] for traj in self.trajectories],
                             [[e.copy() for e in est] for est in self.estimate_trajectories],
                             [[c.copy() for c in cov] for cov in self.covariances]))
        self.slider.setMaximum(self.time_step)
        self.slider.setEnabled(True)
        self.slider.setValue(self.time_step)

    def plot_state(self):
        self.ax.clear()
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_title("2D Target Tracking")
        self.ax.set_xlabel("X position")
        self.ax.set_ylabel("Y position")
        self.ax.grid(True)
        for idx in range(self.num_targets):
            if len(self.trajectories[idx]) > 0:
                traj = np.array(self.trajectories[idx])
                self.ax.plot(traj[:,0], traj[:,1], 'g-', alpha=0.5)
                self.ax.plot(traj[-1,0], traj[-1,1], 'go', label=f"True Target {idx+1}" if idx == 0 else "")
            if len(self.estimate_trajectories[idx]) > 0:
                etraj = np.array(self.estimate_trajectories[idx])
                self.ax.plot(etraj[:,0], etraj[:,1], 'b--', alpha=0.5)
                self.ax.plot(etraj[-1,0], etraj[-1,1], 'bx', label=f"Estimate {idx+1}" if idx == 0 else "")
            # Plot covariance ellipse at estimate
            if self.filter_type in ['KF', 'EKF', 'UKF']:
                kf = self.filters[idx]
                cov = kf.P[0:2,0:2]
                mean = np.array([kf.x[0,0], kf.x[1,0]])
                self.plot_covariance_ellipse(mean, cov, self.ax, n_std=2, alpha=0.2, color='blue')
        self.ax.legend(loc="upper right")
        self.canvas.draw()

    def slider_step(self, step):
        if not self.is_running and 0 <= step < len(self.history):
            trajs, ests, covs = self.history[step]
            self.ax.clear()
            self.ax.set_xlim(0, 100)
            self.ax.set_ylim(0, 100)
            self.ax.set_title(f"2D Target Tracking - Step {step}")
            self.ax.set_xlabel("X position")
            self.ax.set_ylabel("Y position")
            self.ax.grid(True)
            for idx in range(self.num_targets):
                if len(trajs[idx]) > 0:
                    traj = np.array(trajs[idx])
                    self.ax.plot(traj[:,0], traj[:,1], 'g-', alpha=0.5)
                    self.ax.plot(traj[-1,0], traj[-1,1], 'go', label=f"True Target {idx+1}" if idx == 0 else "")
                if len(ests[idx]) > 0:
                    etraj = np.array(ests[idx])
                    self.ax.plot(etraj[:,0], etraj[:,1], 'b--', alpha=0.5)
                    self.ax.plot(etraj[-1,0], etraj[-1,1], 'bx', label=f"Estimate {idx+1}" if idx == 0 else "")
                if len(covs[idx]) > 0:
                    mean = etraj[-1]
                    cov = covs[idx][-1]
                    self.plot_covariance_ellipse(mean, cov, self.ax, n_std=2, alpha=0.2, color='blue')
            self.ax.legend(loc="upper right")
            self.canvas.draw()

    def show_error_heatmap(self):
        import matplotlib.pyplot as plt
        plt.figure("Error Heatmap")
        plt.clf()
        norm = Normalize(vmin=0, vmax=np.max(self.error_map) if np.max(self.error_map) > 0 else 1)
        plt.imshow(self.error_map.T, origin='lower', extent=(0,100,0,100), aspect='auto', cmap=cm.hot, norm=norm)
        plt.colorbar(label='Accumulated Error')
        plt.title('Error Heatmap (where filter made largest errors)')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.show()

    def show_uncertainty_evolution(self):
        import matplotlib.pyplot as plt
        plt.figure("Uncertainty Evolution")
        plt.clf()
        for idx, history in enumerate(self.uncertainty_history):
            plt.plot(history, label=f"Target {idx+1}")
        plt.title('Uncertainty Evolution (Trace of Covariance)')
        plt.xlabel('Time Step')
        plt.ylabel('Trace of Covariance')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KalmanTrackerApp()
    window.show()
    sys.exit(app.exec_())
