"""
Quantum Visualization Module - Visualization tools for quantum states and processes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import seaborn as sns
import logging

logger = logging.getLogger("QuantumViz")

class QuantumVisualizationEngine:
    """Advanced visualization engine for quantum states and processes"""
    
    def __init__(self):
        self.figures = {}
        self.color_schemes = {
            'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            'quantum': ['#00bcd4', '#ff4081', '#651fff', '#00e676', '#ff9100',
                       '#2979ff', '#ff1744', '#1de9b6', '#d500f9', '#ffea00']
        }
        
    def plot_bloch_sphere(self, state, title="Bloch Sphere", save_path=None):
        """Plot a quantum state on the Bloch sphere"""
        if len(state) != 2:
            raise ValueError("Bloch sphere visualization only for single qubits")
        
        # Convert to spherical coordinates
        theta = 2 * np.arccos(np.abs(state[0]))
        phi = np.angle(state[1]) - np.angle(state[0])
        
        # Create Bloch sphere
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_wireframe(x, y, z, color='k', alpha=0.1)
        
        # Plot state vector
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        ax.quiver(0, 0, 0, x, y, z, color='r', arrow_length_ratio=0.1)
        ax.scatter([x], [y], [z], color='r', s=100)
        
        # Add labels
        ax.text(1.1, 0, 0, "|0⟩", fontsize=14)
        ax.text(-1.1, 0, 0, "|1⟩", fontsize=14)
        ax.text(0, 1.1, 0, "|+⟩", fontsize=14)
        ax.text(0, -1.1, 0, "|-⟩", fontsize=14)
        ax.text(0, 0, 1.1, "|0⟩", fontsize=14)
        ax.text(0, 0, -1.1, "|1⟩", fontsize=14)
        
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_zlim([-1.1, 1.1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_probability_distribution(self, state, title="Probability Distribution", save_path=None):
        """Plot the probability distribution of a quantum state"""
        probabilities = np.abs(state) ** 2
        n = len(state)
        num_qubits = int(np.log2(n))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create basis state labels
        labels = [format(i, f'0{num_qubits}b') for i in range(n)]
        
        # Plot probabilities
        bars = ax.bar(range(n), probabilities, color=self.color_schemes['quantum'][:n])
        
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.3f}', ha='center', va='bottom')
        
        ax.set_xlabel('Basis States')
        ax.set_ylabel('Probability')
        ax.set_title(title)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylim(0, 1.1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_density_matrix(self, density_matrix, title="Density Matrix", save_path=None):
        """Plot a density matrix as a heatmap"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot real and imaginary parts separately
        real_part = np.real(density_matrix)
        imag_part = np.imag(density_matrix)
        
        # Create subplots
        if np.any(imag_part != 0):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot real part
            im1 = ax1.imshow(real_part, cmap='RdBu_r', vmin=-1, vmax=1)
            ax1.set_title('Real Part')
            fig.colorbar(im1, ax=ax1)
            
            # Plot imaginary part
            im2 = ax2.imshow(imag_part, cmap='RdBu_r', vmin=-1, vmax=1)
            ax2.set_title('Imaginary Part')
            fig.colorbar(im2, ax=ax2)
            
            fig.suptitle(title)
        else:
            # Only real part
            im = ax.imshow(real_part, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_quantum_circuit(self, gates, qubits, title="Quantum Circuit", save_path=None):
        """Plot a quantum circuit diagram"""
        # This is a simplified implementation
        # In a real application, you would use a proper quantum circuit drawing library
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Draw qubit lines
        for i in range(qubits):
            ax.axhline(y=i, color='k', linestyle='-', alpha=0.3)
            ax.text(-0.5, i, f'Q{i}', fontsize=12, ha='center', va='center')
        
        # Draw gates (simplified)
        for step, gate in enumerate(gates):
            # This would need to be implemented based on the gate format
            pass
        
        ax.set_xlim(-1, len(gates) + 1)
        ax.set_ylim(-0.5, qubits - 0.5)
        ax.set_title(title)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_entanglement_graph(self, graph, title="Entanglement Graph", save_path=None):
        """Plot an entanglement graph"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Draw the graph
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', 
                              node_size=500, ax=ax)
        nx.draw_networkx_edges(graph, pos, edge_color='gray', ax=ax)
        nx.draw_networkx_labels(graph, pos, ax=ax)
        
        ax.set_title(title)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
