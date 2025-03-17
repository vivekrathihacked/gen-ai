"""
Example script to test the performance visualization library with simple test data.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import os
import h5py

# Import our library
from perfviz import BenchmarkVisualizer

def create_test_data(filename="test_benchmark.h5"):
    """
    Create simple test data to demonstrate the visualization library.
    
    This function creates synthetic benchmark data for two algorithms:
    - "fast_algorithm": A new, optimized implementation
    - "slow_algorithm": The baseline implementation
    
    The data simulates performance across different:
    - input_size: Size of the input data (e.g., array length)
    - threads: Number of threads used
    - batch_size: Number of operations in a batch
    """
    # Parameters for our benchmark
    input_sizes = [10, 50, 100, 500, 1000]
    threads = [1, 2, 4, 8]
    batch_sizes = [1, 10, 100]
    
    # Create empty dataframes to store our results
    fast_algorithm_results = []
    slow_algorithm_results = []
    
    # Generate synthetic benchmark data
    np.random.seed(42)  # For reproducibility
    
    for m in input_sizes:
        for p in threads:
            for batch in batch_sizes:
                # Base computation time (arbitrary formula)
                base_time = m * (1.0 / p) * (0.1 + 0.9 * batch / 100)
                
                # Add some noise to make it more realistic
                noise_factor = 0.1
                
                # Fast algorithm is generally faster, but the advantage varies
                # based on input size and threads
                speedup = 1.0 + (m / 1000) * (p / 8)  # Larger inputs benefit more
                
                # Slow algorithm time
                slow_time = base_time * (1 + np.random.normal(0, noise_factor))
                
                # Fast algorithm time
                fast_time = (base_time / speedup) * (1 + np.random.normal(0, noise_factor))
                
                # Create row for each algorithm
                slow_row = {
                    'm': m,           # Input size
                    'p': p,           # Thread count
                    'n': batch,       # Batch size (using 'n' to match original code)
                    'batch_size': 1,  # Fixed to 1 for simplicity
                    'time': slow_time
                }
                
                fast_row = {
                    'm': m,
                    'p': p,
                    'n': batch,
                    'batch_size': 1,
                    'time': fast_time
                }
                
                slow_algorithm_results.append(slow_row)
                fast_algorithm_results.append(fast_row)
    
    # Convert to dataframes
    df_slow = pd.DataFrame(slow_algorithm_results).set_index(['m', 'p', 'n', 'batch_size'])
    df_fast = pd.DataFrame(fast_algorithm_results).set_index(['m', 'p', 'n', 'batch_size'])
    
    # Save to HDF5 file
    with pd.HDFStore(filename, mode='w') as store:
        store.put('data_slow_algorithm', df_slow)
        store.put('data_fast_algorithm', df_fast)
    
    print(f"Test data created and saved to {filename}")
    return filename

def run_visualization():
    """Run the performance visualization on our test data."""
    # First, ensure we have test data
    data_file = "test_benchmark.h5"
    if not os.path.exists(data_file):
        data_file = create_test_data(data_file)
    
    # Create output directory
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize our visualizer
    visualizer = BenchmarkVisualizer(data_file, output_dir)
    
    # Load data comparing slow vs fast algorithms
    # We use slow_algorithm as reference and compare fast_algorithm against it
    dataframes = visualizer.load_data(
        reference_key="slow_algorithm",
        comparison_keys=["fast_algorithm"]
    )
    
    # Prepare data for visualization
    df = visualizer.prepare_data(
        dataframes["fast_algorithm"], 
        dimensions=["m", "p", "n", "batch_size"]
    )
    
    # Create visualization
    fig, axes = visualizer.create_heatmap(
        df, 
        dimensions=["m", "p", "n", "batch_size"],
        output_file="algorithm_speedup.pdf",
        title="Performance Improvement: Fast Algorithm vs Slow Algorithm",
        vmin=0.5,
        vmax=5.0  # Adjust based on your generated data
    )
    
    print(f"Visualization saved to {output_dir}/algorithm_speedup.pdf")
    
    # Return the figure for interactive usage
    return fig, axes

if __name__ == "__main__":
    run_visualization()