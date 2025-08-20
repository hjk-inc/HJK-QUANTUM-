"""
Quantum Processing Playground - Executive Module
Main entry point that integrates all quantum processing components and provides
a unified interface for quantum algorithm development and execution.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from scipy.linalg import expm, norm
from scipy.optimize import minimize
from scipy.stats import entropy
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import torch
import torch.nn as nn
import qutip as qt
import datetime
import time
import logging
import json
import sys
import os

# Import our custom modules
from quantum_core import QuantumStateManager, QuantumGateLibrary, QuantumAlgorithmSuite
from quantum_viz import QuantumVisualizationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantum_playground.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumPlayground")

class QuantumPlayground:
    """
    Main playground class that integrates all quantum processing components
    and provides a unified interface for quantum algorithm development.
    """
    
    def __init__(self, num_qubits=4, enable_gpu=False):
        self.num_qubits = num_qubits
        self.enable_gpu = enable_gpu
        self.initialized = False
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.initialize_components()
        
        logger.info(f"Quantum Playground initialized with session ID: {self.session_id}")
    
    def initialize_components(self):
        """Initialize all quantum processing components"""
        try:
            # Core quantum components
            self.state_manager = QuantumStateManager(self.num_qubits)
            self.gate_library = QuantumGateLibrary()
            self.algorithm_suite = QuantumAlgorithmSuite(self.state_manager, self.gate_library)
            
            # Visualization engine
            self.viz_engine = QuantumVisualizationEngine()
            
            # Session data
            self.session_data = {
                'start_time': datetime.datetime.now(),
                'operations': [],
                'results': {},
                'config': {
                    'num_qubits': self.num_qubits,
                    'enable_gpu': self.enable_gpu
                }
            }
            
            self.initialized = True
            logger.info("All quantum components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize quantum components: {str(e)}")
            raise
    
    def run_algorithm(self, algorithm_name, **kwargs):
        """
        Execute a quantum algorithm by name with provided parameters
        
        Args:
            algorithm_name: Name of the algorithm to run
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Dictionary with algorithm results
        """
        if not self.initialized:
            raise RuntimeError("Quantum playground not initialized")
        
        if algorithm_name not in self.algorithm_suite.algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not available")
        
        start_time = time.time()
        
        try:
            # Execute the algorithm
            result = self.algorithm_suite.algorithms[algorithm_name](**kwargs)
            
            # Record the operation
            operation_record = {
                'timestamp': datetime.datetime.now(),
                'algorithm': algorithm_name,
                'parameters': kwargs,
                'execution_time': time.time() - start_time,
                'result_keys': list(result.keys()) if isinstance(result, dict) else 'N/A'
            }
            
            self.session_data['operations'].append(operation_record)
            
            # Store the result with a unique key
            result_key = f"{algorithm_name}_{len(self.session_data['operations'])}"
            self.session_data['results'][result_key] = result
            
            logger.info(f"Algorithm {algorithm_name} executed successfully in {operation_record['execution_time']:.4f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Error executing algorithm {algorithm_name}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def create_custom_gate(self, name, matrix):
        """
        Create a custom quantum gate
        
        Args:
            name: Name for the custom gate
            matrix: Unitary matrix representing the gate
            
        Returns:
            Success confirmation
        """
        try:
            self.gate_library.add_custom_gate(name, matrix)
            logger.info(f"Custom gate {name} created successfully")
            return {"status": "success", "gate_name": name}
        except Exception as e:
            error_msg = f"Error creating custom gate: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def visualize_state(self, state_name, visualization_type="bloch", **kwargs):
        """
        Visualize a quantum state using the specified visualization type
        
        Args:
            state_name: Name of the state to visualize
            visualization_type: Type of visualization to create
            **kwargs: Additional visualization parameters
            
        Returns:
            Visualization object or file path
        """
        if state_name not in self.state_manager.states:
            raise ValueError(f"State {state_name} not found")
        
        state = self.state_manager.states[state_name]
        
        if visualization_type == "bloch":
            return self.viz_engine.plot_bloch_sphere(state, title=f"State: {state_name}", **kwargs)
        elif visualization_type == "probability":
            return self.viz_engine.plot_probability_distribution(state, title=f"State: {state_name}", **kwargs)
        elif visualization_type == "density":
            density_matrix = np.outer(state, np.conj(state))
            return self.viz_engine.plot_density_matrix(density_matrix, title=f"State: {state_name}", **kwargs)
        else:
            raise ValueError(f"Visualization type {visualization_type} not supported")
    
    def quantum_machine_learning(self, data, labels, algorithm="qsvm", **kwargs):
        """
        Execute quantum machine learning algorithms
        
        Args:
            data: Input data
            labels: Target labels
            algorithm: Type of quantum ML algorithm
            **kwargs: Algorithm-specific parameters
            
        Returns:
            ML model and results
        """
        if algorithm == "qsvm":
            return self.algorithm_suite.quantum_support_vector_machine(data, labels, **kwargs)
        elif algorithm == "qnn":
            return self.algorithm_suite.quantum_neural_network(data, labels, **kwargs)
        elif algorithm == "qcluster":
            return self.algorithm_suite.quantum_clustering(data, **kwargs)
        else:
            raise ValueError(f"Quantum ML algorithm {algorithm} not supported")
    
    def run_benchmark(self, benchmark_type="entanglement", qubits_range=None, iterations=10):
        """
        Run performance benchmarks for quantum operations
        
        Args:
            benchmark_type: Type of benchmark to run
            qubits_range: Range of qubit numbers to test
            iterations: Number of iterations per test
            
        Returns:
            Benchmark results
        """
        if qubits_range is None:
            qubits_range = range(2, 8)
        
        results = {
            'benchmark_type': benchmark_type,
            'qubits_range': list(qubits_range),
            'iterations': iterations,
            'timings': {},
            'memory_usage': {}
        }
        
        for n_qubits in qubits_range:
            timings = []
            memory_usage = []
            
            for i in range(iterations):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                if benchmark_type == "entanglement":
                    # Create and measure entangled states
                    state_name = f"bell_state_{n_qubits}_{i}"
                    self.state_manager.create_state(state_name, 
                                                   np.ones(2**n_qubits) / np.sqrt(2**n_qubits))
                    self.state_manager.measure(state_name, shots=1000)
                
                elif benchmark_type == "gate_operations":
                    # Apply multiple gates
                    for q in range(n_qubits):
                        self.state_manager.apply_gate(f"q{q}_zero", self.gate_library.get_gate('H'))
                
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                timings.append(end_time - start_time)
                memory_usage.append(end_memory - start_memory)
            
            results['timings'][n_qubits] = {
                'mean': np.mean(timings),
                'std': np.std(timings),
                'min': np.min(timings),
                'max': np.max(timings)
            }
            
            results['memory_usage'][n_qubits] = {
                'mean': np.mean(memory_usage),
                'std': np.std(memory_usage),
                'min': np.min(memory_usage),
                'max': np.max(memory_usage)
            }
        
        return results
    
    def _get_memory_usage(self):
        """Get current memory usage (simplified implementation)"""
        # This is a simplified implementation - in production, use psutil or similar
        return 0
    
    def save_session(self, filename=None):
        """
        Save the current session to a file
        
        Args:
            filename: Name of the file to save to
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = f"quantum_session_{self.session_id}.json"
        
        try:
            # Convert session data to JSON-serializable format
            session_copy = self.session_data.copy()
            session_copy['start_time'] = session_copy['start_time'].isoformat()
            
            for op in session_copy['operations']:
                op['timestamp'] = op['timestamp'].isoformat()
            
            with open(filename, 'w') as f:
                json.dump(session_copy, f, indent=2)
            
            logger.info(f"Session saved to {filename}")
            return filename
            
        except Exception as e:
            error_msg = f"Error saving session: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def load_session(self, filename):
        """
        Load a session from a file
        
        Args:
            filename: Name of the file to load from
            
        Returns:
            Loaded session data
        """
        try:
            with open(filename, 'r') as f:
                session_data = json.load(f)
            
            # Convert string timestamps back to datetime objects
            session_data['start_time'] = datetime.datetime.fromisoformat(session_data['start_time'])
            
            for op in session_data['operations']:
                op['timestamp'] = datetime.datetime.fromisoformat(op['timestamp'])
            
            self.session_data = session_data
            logger.info(f"Session loaded from {filename}")
            return session_data
            
        except Exception as e:
            error_msg = f"Error loading session: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_session_summary(self):
        """
        Get a summary of the current session
        
        Returns:
            Dictionary with session summary
        """
        if not self.initialized:
            return {"status": "not_initialized"}
        
        return {
            "session_id": self.session_id,
            "start_time": self.session_data['start_time'],
            "duration": str(datetime.datetime.now() - self.session_data['start_time']),
            "operations_count": len(self.session_data['operations']),
            "results_count": len(self.session_data['results']),
            "config": self.session_data['config']
        }
    
    def export_results(self, format_type="json", filename=None):
        """
        Export results in the specified format
        
        Args:
            format_type: Format to export to (json, csv, etc.)
            filename: Name of the output file
            
        Returns:
            Path to the exported file
        """
        if filename is None:
            filename = f"quantum_results_{self.session_id}.{format_type}"
        
        try:
            if format_type == "json":
                with open(filename, 'w') as f:
                    json.dump(self.session_data['results'], f, indent=2, default=str)
            
            elif format_type == "csv":
                # Flatten results for CSV export
                flattened_data = []
                for key, result in self.session_data['results'].items():
                    if isinstance(result, dict):
                        row = {"result_key": key}
                        row.update(result)
                        flattened_data.append(row)
                
                if flattened_data:
                    df = pd.DataFrame(flattened_data)
                    df.to_csv(filename, index=False)
                else:
                    # Create empty CSV with just headers
                    pd.DataFrame().to_csv(filename, index=False)
            
            else:
                raise ValueError(f"Export format {format_type} not supported")
            
            logger.info(f"Results exported to {filename} in {format_type} format")
            return filename
            
        except Exception as e:
            error_msg = f"Error exporting results: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

# Example usage and demonstration
def demonstrate_playground():
    """Demonstrate the capabilities of the Quantum Playground"""
    
    print("Initializing Quantum Playground...")
    playground = QuantumPlayground(num_qubits=4)
    
    print("\n1. Creating and visualizing quantum states...")
    # Create a custom quantum state
    custom_state = np.array([0.6, 0.8j], dtype=np.complex128)
    playground.state_manager.create_state("custom_state", custom_state)
    
    # Visualize the state on Bloch sphere
    fig = playground.visualize_state("custom_state", "bloch")
    plt.savefig("custom_state_bloch.png")
    plt.close()
    
    print("\n2. Running quantum algorithms...")
    # Run Grover's search algorithm (simplified demonstration)
    oracle = np.eye(8)  # Identity matrix as placeholder
    grover_result = playground.run_algorithm("grover", oracle=oracle, num_qubits=3, iterations=2)
    print(f"Grover's search result: {grover_result}")
    
    print("\n3. Quantum machine learning demonstration...")
    # Generate sample data
    np.random.seed(42)
    data = np.random.randn(20, 4)  # 20 samples, 4 features
    labels = np.random.randint(0, 2, 20)  # Binary labels
    
    # Run quantum SVM
    qsvm_result = playground.quantum_machine_learning(data, labels, "qsvm")
    print(f"Quantum SVM accuracy: {qsvm_result['accuracy']:.4f}")
    
    print("\n4. Running benchmarks...")
    benchmark_result = playground.run_benchmark("entanglement", qubits_range=[2, 3, 4], iterations=3)
    print(f"Benchmark results: {benchmark_result}")
    
    print("\n5. Saving session...")
    session_file = playground.save_session()
    print(f"Session saved to {session_file}")
    
    print("\n6. Generating session summary...")
    summary = playground.get_session_summary()
    print(f"Session summary: {summary}")
    
    print("\n7. Exporting results...")
    results_file = playground.export_results("json")
    print(f"Results exported to {results_file}")
    
    print("\nQuantum Playground demonstration completed successfully!")
    return playground

# Command-line interface
def main():
    """Main command-line interface for the Quantum Playground"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantum Processing Playground")
    parser.add_argument("--qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--algorithm", type=str, help="Algorithm to run")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--visualize", type=str, help="State to visualize")
    
    args = parser.parse_args()
    
    if args.demo:
        playground = demonstrate_playground()
    elif args.algorithm:
        playground = QuantumPlayground(num_qubits=args.qubits)
        # Here you would add code to run the specified algorithm
        print(f"Running algorithm: {args.algorithm}")
    elif args.benchmark:
        playground = QuantumPlayground(num_qubits=args.qubits)
        results = playground.run_benchmark()
        print(json.dumps(results, indent=2))
    elif args.visualize:
        playground = QuantumPlayground(num_qubits=args.qubits)
        fig = playground.visualize_state(args.visualize, "bloch")
        plt.show()
    else:
        print("Quantum Processing Playground initialized. Use --help for options.")

if __name__ == "__main__":
    main()
