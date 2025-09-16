# 🌀 Quantum Machine Learning with Variational Quantum Classifier

This project demonstrates the implementation of a **Variational Quantum Classifier (VQC)** for binary classification tasks using quantum circuits. The classifier is applied to a synthetic "moons" dataset, which features two interlocking half-circles, presenting a non-linearly separable challenge. By encoding classical data into quantum states and optimizing a parameterized quantum ansatz, the model learns a decision boundary through hybrid quantum-classical optimization. This approach highlights the potential of quantum machine learning in handling complex data patterns via superposition and entanglement.

The codebase integrates Qiskit for quantum circuit design and simulation, NumPy for numerical computations, Matplotlib for visualization, and scikit-learn for data preprocessing and evaluation metrics.

---

## 📂 Code Structure
The project is organized as a single, self-contained Python script that encapsulates the entire VQC pipeline. Below is a detailed breakdown of the modular components:

- **Dataset Generation and Preprocessing** (Lines 20-35): Generates the moons dataset, normalizes features to the interval [0, π] for quantum encoding compatibility, and performs a train-test split. Includes visualization of the training data in the scaled feature space.
  
- **Feature Encoding Circuit** (Lines 38-48): Defines the `feature_map` function, which uses angle encoding via RY and RZ rotations to map classical feature vectors into quantum states on the specified number of qubits.

- **Variational Ansatz Circuit** (Lines 51-65): Implements the `variational_ansatz` function, constructing a layered quantum circuit with rotation (RY, RZ) and entangling (CNOT) gates. The depth is fixed at 3 layers for expressivity while maintaining computational feasibility.

- **Full VQC Circuit Assembly** (Lines 68-76): Combines the feature map and ansatz in the `vqc_circuit` function, appending measurements to all qubits for expectation value computation.

- **Circuit Execution and Expectation Value** (Lines 79-97): Sets up the QASM simulator backend and defines `circuit_expectation`, which executes the circuit, retrieves counts, and computes the expectation value of the Pauli-Z operator on the first qubit.

- **Cost Function Definition** (Lines 100-108): Formulates the `cost_function` as the mean squared error between predicted expectation values (mapped to labels) and true labels, enabling gradient-free optimization.

- **Training Loop** (Lines 111-122): Initializes random parameters, employs the COBYLA optimizer for 50 iterations across 20 outer loops, and monitors loss convergence during training.

- **Prediction and Evaluation** (Lines 125-133): Generates predictions on the test set by thresholding expectation values and computes classification accuracy using scikit-learn.

- **Decision Boundary Visualization** (Lines 136-149): Creates a mesh grid over the feature space, evaluates the classifier on grid points, and plots the resulting decision boundary overlaid with test data points.

---

## 🔑 Important Variables
Key variables are defined throughout the script to control dataset properties, quantum circuit parameters, and optimization settings. Precise tuning of these can influence model performance and simulation efficiency:

- `n_samples=200` → Total number of data points in the moons dataset; increasing this enhances training robustness but raises computational demands.
  
- `noise=0.2` → Gaussian noise level added to the dataset; higher values simulate real-world data imperfections, testing the classifier's generalization.
  
- `X_scaled` → Normalized feature matrix in [0, π]; this scaling ensures rotational gates operate within valid quantum parameter ranges.
  
- `test_size=0.3` → Fraction of data reserved for testing; a 70/30 split balances training data availability with reliable evaluation.
  
- `num_qubits=2` → Number of qubits, matching the dataset's feature dimensionality; scalable for higher-dimensional problems.
  
- `depth=3` → Layers in the variational ansatz; deeper circuits increase expressivity but risk barren plateaus in optimization.
  
- `num_params = X_train.shape[1] * 3 * 2` → Total variational parameters (12 for this setup); calculated as features × depth × gates per layer.
  
- `shots=1024` → Measurement shots per circuit execution; more shots reduce statistical noise in expectation values at the cost of runtime.
  
- `maxiter=50` → Maximum iterations per COBYLA optimization step; adjustable to balance convergence speed and solution quality.
  
- `optimizer=COBYLA()` → Derivative-free optimizer; suitable for noisy quantum landscapes, with alternatives like SPSA for larger scales.

---

## ⚙️ How to Interact
To engage with the project, follow these structured steps. Ensure a Python environment (version 3.8+) is configured with the required dependencies for seamless execution.

1. **Install Dependencies**: Execute the following command in your terminal to install necessary packages:
   ```bash
   pip install qiskit numpy matplotlib scikit-learn
   ```
   These libraries provide quantum simulation, numerical operations, plotting, and machine learning utilities, respectively.

2. **Prepare the Script**: Save the provided code as `vqc.py`. Review and customize variables (e.g., increase `n_samples` to 500 for denser data or adjust `shots` to 2048 for finer-grained measurements).

3. **Run the Full Pipeline**: Launch the script to generate the dataset, train the model, evaluate accuracy, and display visualizations:
   ```bash
   python vqc.py
   ```
   Monitor console output for iteration-wise loss values and final test accuracy.

4. **Experiment with Hyperparameters**: Modify `depth` to 5 for a more expressive ansatz or switch the optimizer to `SLSQP` (import from `qiskit.algorithms.optimizers`) for potentially faster convergence. Re-run to observe impacts on accuracy and training time.

5. **Extend for Real Hardware**: Replace `Aer.get_backend("qasm_simulator")` with a real backend via `IBMProvider` (after obtaining an IBM Quantum account) to test on actual quantum devices, noting increased noise and execution delays.

6. **Troubleshoot Common Issues**: If simulations are slow, reduce `shots` or `n_samples`. For plotting errors, ensure Matplotlib's backend supports interactive displays (e.g., via `%matplotlib inline` in Jupyter).

---

## 🧠 Physical/Statistical Intuition
The VQC bridges classical machine learning with quantum principles, offering insights into hybrid algorithms. Each component draws from established quantum information theory and statistical mechanics:

- **Quantum Feature Encoding**: Classical features are transformed into quantum amplitudes via single-qubit rotations (RY for real parts, RZ for phases). This embeds data into a high-dimensional Hilbert space, potentially revealing non-local correlations absent in classical embeddings.

- **Variational Principle**: Inspired by the Rayleigh-Ritz method in quantum mechanics, the ansatz approximates the ground state of a data-dependent Hamiltonian. Parameters are varied to minimize energy (here, classification loss), akin to finding the lowest eigenvalue.

- **Entanglement in Ansatz Layers**: CNOT gates induce qubit correlations, enabling the circuit to capture entangled representations. This mirrors how quantum systems exploit superposition for exponential state spaces, aiding in non-linear decision boundaries.

- **Expectation Value as Predictor**: The <Z> observable on the first qubit serves as a binary discriminator; positive values indicate one class, negative the other. Statistically, this is a probabilistic readout, with variance scaling as 1/√shots due to the central limit theorem.

- **Hybrid Optimization Landscape**: The cost function forms a non-convex surface over parameter space, complicated by quantum shot noise. COBYLA navigates this via constraint satisfaction, reflecting the "noisy intermediate-scale quantum" (NISQ) era's reliance on classical feedback loops.

- **Generalization and Overfitting**: The moons dataset tests separability; quantum models may generalize better for quantum-native data but require regularization (e.g., shallower depths) to avoid overfitting in classical-like tasks.

- **Scaling Behavior**: As qubit count grows, circuit depth must be controlled to mitigate error accumulation, underscoring the trade-off between expressivity and trainability in quantum neural networks.

---

## 🧮 Numerical Models
This implementation employs a suite of established techniques from quantum computing and machine learning, ensuring reproducibility and extensibility:

- **Synthetic Dataset Generation**: Utilizes scikit-learn's `make_moons` for a benchmark non-linear binary classification problem, with added Gaussian noise to mimic empirical datasets.

- **Min-Max Normalization**: Scales features to [0, π] via `MinMaxScaler`, aligning with the periodic nature of quantum rotation gates and preventing overflow in circuit parameters.

- **Angle Encoding Scheme**: A compact, hardware-efficient encoding using RY/RZ pairs per feature, contrasting with amplitude encoding which requires ancillary qubits for state preparation.

- **Layered Variational Ansatz**: A hardware-ansatz inspired by quantum approximate optimization algorithms (QAOA), featuring alternating unitary rotations and entangling blocks for universal approximation capabilities.

- **Pauli-Z Observable Measurement**: Computes class scores via partial tomography on the first qubit, reducing measurement overhead compared to full state tomography.

- **Mean Squared Error Cost**: A differentiable surrogate for binary cross-entropy, facilitating classical optimization; alternatives like fidelity-based losses could enhance quantum-specific metrics.

- **COBYLA Optimization**: A constrained, derivative-free method from SciPy, robust to the black-box nature of quantum evaluations; supports up to 100 variables efficiently.

- **Monte Carlo Sampling**: Circuit executions use shot-based sampling for expectation values, introducing beneficial stochasticity that aids escape from local minima.

- **Mesh-Grid Decision Boundary**: Employs NumPy's `meshgrid` and `contourf` for 2D visualization, enabling qualitative assessment of the learned hypersurface in the feature plane.

- **Accuracy Metric**: Leverages scikit-learn's `accuracy_score` for quantitative evaluation, with potential extensions to ROC-AUC for imbalanced datasets.
