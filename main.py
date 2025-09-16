# ==============================================================
# Quantum Machine Learning with Variational Quantum Classifier
# Dataset: Binary classification on a toy dataset (circles / moons)
# Framework: Qiskit + NumPy + Matplotlib
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.utils import QuantumInstance
from qiskit.opflow import Z, I
from qiskit.algorithms.optimizers import COBYLA

# --------------------------------------------------------------
# 1. Generate and preprocess dataset
# --------------------------------------------------------------

# Make dataset
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# Normalize features to [0, pi]
scaler = MinMaxScaler((0, np.pi))
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

plt.figure(figsize=(6,6))
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap="coolwarm")
plt.title("Training Data (Quantum Feature Space)")
plt.show()

# --------------------------------------------------------------
# 2. Define feature encoding circuit (Angle Encoding)
# --------------------------------------------------------------

def feature_map(x):
    """
    Encode classical data vector x into quantum state amplitudes.
    Using angle encoding with RX and RY rotations.
    """
    num_qubits = len(x)
    qc = QuantumCircuit(num_qubits)
    for i, val in enumerate(x):
        qc.ry(val, i)
        qc.rz(val, i)
    return qc

# --------------------------------------------------------------
# 3. Define variational ansatz circuit
# --------------------------------------------------------------

def variational_ansatz(num_qubits, params):
    """
    Build a parameterized quantum circuit with entanglement.
    params: list of parameters (length = num_qubits * depth * 2)
    """
    depth = 3
    qc = QuantumCircuit(num_qubits)
    param_idx = 0
    for d in range(depth):
        # Rotation layer
        for q in range(num_qubits):
            qc.ry(params[param_idx], q)
            param_idx += 1
            qc.rz(params[param_idx], q)
            param_idx += 1
        # Entangling layer
        for q in range(num_qubits - 1):
            qc.cx(q, q+1)
    return qc

# --------------------------------------------------------------
# 4. Build full VQC circuit
# --------------------------------------------------------------

def vqc_circuit(x, params):
    """
    Combine feature map + variational ansatz.
    """
    num_qubits = len(x)
    qc = feature_map(x)
    qc = qc.compose(variational_ansatz(num_qubits, params))
    qc.measure_all()
    return qc

# --------------------------------------------------------------
# 5. Define expectation value function
# --------------------------------------------------------------

backend = Aer.get_backend("qasm_simulator")
shots = 1024

def circuit_expectation(x, params):
    """
    Execute circuit and compute expectation value of Z⊗I observable.
    """
    qc = feature_map(x)
    qc = qc.compose(variational_ansatz(len(x), params))
    qc.save_statevector()
    result = execute(qc, backend, shots=shots).result()
    counts = result.get_counts()
    
    # Observable = Z on first qubit
    exp_val = 0
    for bitstring, count in counts.items():
        if bitstring[-1] == '0':
            exp_val += count
        else:
            exp_val -= count
    return exp_val / shots

# --------------------------------------------------------------
# 6. Define cost function
# --------------------------------------------------------------

def cost_function(params, X, y):
    loss = 0
    for xi, yi in zip(X, y):
        prediction = circuit_expectation(xi, params)
        loss += (prediction - (1 - 2*yi))**2
    return loss / len(X)

# --------------------------------------------------------------
# 7. Train the Variational Quantum Classifier
# --------------------------------------------------------------

num_params = X_train.shape[1] * 3 * 2  # heuristic parameter size
params = np.random.uniform(0, 2*np.pi, num_params)

optimizer = COBYLA(maxiter=50)

for it in range(20):
    loss = cost_function(params, X_train, y_train)
    print(f"Iteration {it} | Loss: {loss:.4f}")
    params = optimizer.optimize(
        num_vars=len(params),
        objective_function=lambda p: cost_function(p, X_train, y_train),
        initial_point=params
    )[0]

# --------------------------------------------------------------
# 8. Test Accuracy
# --------------------------------------------------------------

y_pred = []
for xi in X_test:
    val = circuit_expectation(xi, params)
    pred = 0 if val > 0 else 1
    y_pred.append(pred)

acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)

# --------------------------------------------------------------
# 9. Visualize Decision Boundary
# --------------------------------------------------------------

xx, yy = np.meshgrid(
    np.linspace(0, np.pi, 50),
    np.linspace(0, np.pi, 50)
)
Zs = []
for (i, j) in zip(xx.ravel(), yy.ravel()):
    val = circuit_expectation([i, j], params)
    Zs.append(0 if val > 0 else 1)
Zs = np.array(Zs).reshape(xx.shape)

plt.figure(figsize=(6,6))
plt.contourf(xx, yy, Zs, cmap="coolwarm", alpha=0.6)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap="coolwarm", edgecolors="k")
plt.title("Decision Boundary Learned by Variational Quantum Classifier")
plt.show()
