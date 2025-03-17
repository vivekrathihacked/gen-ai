"""
perfviz - A library for visualizing performance benchmarks of numerical operations
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from itertools import product
from matplotlib.colors import LinearSegmentedColormap, LogNorm, hsv_to_rgb
from matplotlib.ticker import FuncFormatter, FixedLocator
import os

class BenchmarkVisualizer:
    """
    Visualize performance benchmarks comparing different implementations
    of numerical operations.
    """

    def __init__(self, data_file, output_dir='output'):
        """
        Initialize the visualizer.
        
        Parameters:
        - data_file: Path to the HDF5 file containing benchmark data
        - output_dir: Directory to save output visualizations
        """
        self.data_file = Path(data_file)
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        matplotlib.use('agg')
        
    def load_data(self, reference_key, comparison_keys, invert=False):
        """
        Load benchmark data from HDF5 file.
        
        Parameters:
        - reference_key: Key for the reference dataset
        - comparison_keys: List of keys for datasets to compare against the reference
        - invert: If True, inverts the performance ratio calculation
        
        Returns:
        - Dictionary of loaded dataframes
        """
        self.dataframes = {}
        
        # Load all datasets
        for key in [reference_key] + comparison_keys:
            try:
                self.dataframes[key] = pd.read_hdf(self.data_file, f'data_{key}')
            except KeyError:
                print(f"Warning: Key 'data_{key}' not found in {self.data_file}")
        
        # Calculate performance ratios
        reference_df = self.dataframes[reference_key]
        
        for key in comparison_keys:
            if key in self.dataframes:
                comparison_df = self.dataframes[key]
                # Make sure we're comparing the same rows
                comparison_df = comparison_df.loc[reference_df.index]
                
                # Calculate speedup ratio
                if invert:
                    comparison_df['delta'] = reference_df['time'] / comparison_df['time']
                else:
                    comparison_df['delta'] = comparison_df['time'] / reference_df['time']
                
                self.dataframes[key] = comparison_df
        
        return self.dataframes
    
    def prepare_data(self, df, dimensions, filters=None):
        """
        Prepare data for visualization.
        
        Parameters:
        - df: DataFrame containing benchmark results
        - dimensions: List of dimension names to use for visualization
        - filters: Dictionary of filters to apply to the data
        
        Returns:
        - Prepared DataFrame
        """
        # Reset index for easier manipulation
        df = df.reset_index()
        
        # Sort by dimensions
        df.sort_values(by=dimensions, inplace=True)
        
        # Apply filters if provided
        if filters:
            for column, condition in filters.items():
                if callable(condition):
                    df = df[condition(df[column])]
                else:
                    df = df[df[column] == condition]
        
        return df
    
    def create_heatmap(self, df, dimensions, output_file=None, 
                      value_column='delta', vmin=0.5, vmax=100,
                      title=None, cmap=None):
        """
        Create a heatmap visualization of performance ratios.
        
        Parameters:
        - df: DataFrame containing benchmark results
        - dimensions: List of 4 dimension names [x_axis, y_axis, row, column]
        - output_file: Path to save the output visualization
        - value_column: Column containing the values to visualize
        - vmin, vmax: Minimum and maximum values for the color scale
        - title: Title for the visualization
        - cmap: Custom colormap (if None, a default one will be created)
        
        Returns:
        - Figure and axes objects
        """
        if len(dimensions) != 4:
            raise ValueError("Four dimensions required: [x_axis, y_axis, row, column]")
            
        # Extract dimension values
        dim_name = dimensions
        dim_value = [df[name].unique() for name in dim_name]
        
        rows = len(dim_value[2])
        columns = len(dim_value[3])
        
        # Create custom colormap if not provided
        if cmap is None:
            cmap = self._create_default_colormap(vmin, vmax)
        
        # Create figure and axes
        fig, axes = plt.subplots(
            rows, columns, figsize=(4*columns, 2*rows), sharex=True, sharey=True
        )
        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        
        # If there's only one row or column, axes won't be a 2D array
        if rows == 1 and columns == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif columns == 1:
            axes = axes.reshape(-1, 1)
        
        # Create heatmaps
        norm = LogNorm(vmin=vmin, vmax=vmax)
        
        for (i, dim2), (j, dim3) in product(
            enumerate(dim_value[2]), enumerate(dim_value[3])
        ):
            ax = axes[i, j]
            
            # Filter data for this subplot
            filtered_data = df.loc[
                (df[dim_name[2]] == dim2) & (df[dim_name[3]] == dim3)
            ]
            
            # Create pivot table
            try:
                pivot_data = filtered_data.pivot(
                    index=dim_name[0], columns=dim_name[1], values=value_column
                )
                
                # Create heatmap
                heatmap = ax.imshow(
                    pivot_data,
                    extent=(
                        min(dim_value[0]), max(dim_value[0]),
                        min(dim_value[1]), max(dim_value[1])
                    ),
                    origin='lower',
                    aspect='auto',
                    cmap=cmap,
                    norm=norm
                )
                
                # Set labels
                if i == 0:
                    ax.set_title(f"{dim_name[3]}={dim3}")
                if j == 0:
                    ax.set_ylabel(f"{dim_name[2]}={dim2}")
                if i == rows - 1:
                    ax.set_xlabel(dim_name[0])
                if j == columns - 1 and i == rows - 1:
                    ax.set_xlabel(f"{dim_name[1]}       {dim_name[0]}")
            
            except ValueError as e:
                print(f"Warning: Could not create heatmap for {dim2},{dim3}: {e}")
        
        # Add colorbar
        plt.tight_layout()
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(heatmap, cax=cbar_ax)
        cbar.set_label("Speedup Ratio")
        
        # Format colorbar ticks
        self._format_colorbar(cbar, vmin, vmax)
        
        # Set overall title
        if title:
            fig.suptitle(title, fontsize=16)
        
        if output_file:
            output_path = self.output_dir / output_file
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        
        return fig, axes
    
    def _create_default_colormap(self, vmin, vmax):
        """Create a default colormap for performance visualization."""
        norm = LogNorm(vmin=vmin, vmax=vmax)
        
        # Define colormap points
        yellow_begin = 1.0
        yellow_end = 1.10
        green = 2
        cyan = 5
        blue = 10
        
        # HSV to RGB conversion for better color control
        make_color = lambda x: hsv_to_rgb((x/6, 1, 1))
        
        # Create custom colormap
        custom_cmap = LinearSegmentedColormap.from_list("performance_cmap", [
            (0.0, make_color(0)),  # Red for slowdown
            (norm(yellow_begin), make_color(1)),  # Yellow for neutral
            (norm(yellow_end), make_color(1)),
            (norm(green), make_color(2)),  # Green for modest speedup
            (norm(cyan), make_color(3)),  # Cyan for good speedup
            (norm(blue), make_color(4)),  # Blue for great speedup
            (1, make_color(5)),  # Purple for excellent speedup
        ], N=1000)
        
        return custom_cmap
    
    def _format_colorbar(self, cbar, vmin, vmax):
        """Format the colorbar with appropriate ticks and labels."""
        def log_format(x, pos):
            return f"{int(x)}" if x >= 1 else f"{x:.1f}"
        
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(log_format))
        
        # Set reasonable tick positions
        tick_values = [vmin]
        
        # Add 0.5, 0.75 if vmin is low enough
        if vmin <= 0.5:
            tick_values.extend([0.5, 0.75])
            
        # Add standard values
        tick_values.extend([1, 2, 5, 10, 20, 50])
        
        # Add vmax if it's not already in the list
        if vmax not in tick_values:
            tick_values.append(vmax)
            
        # Filter out values outside the range
        tick_values = [v for v in tick_values if vmin <= v <= vmax]
        
        cbar.ax.yaxis.set_major_locator(FixedLocator(tick_values))


# Example usage
if __name__ == "__main__":
    # Example usage of the library
    visualizer = BenchmarkVisualizer("benchmark_results.h5", "visualizations")
    
    # Load data comparing vanilla vs patched implementations
    dataframes = visualizer.load_data(
        reference_key="patched_strided",
        comparison_keys=["vanilla_strided", "vanilla_continuous", "patched_continuous"]
    )
    
    # Prepare data for visualization
    df = visualizer.prepare_data(
        dataframes["vanilla_strided"], 
        dimensions=["m", "p", "n", "batch_size"],
        filters={"m": lambda x: x != 1, "p": lambda x: x != 1, "n": lambda x: x != 1}
    )
    
    # Create visualization
    visualizer.create_heatmap(
        df, 
        dimensions=["m", "p", "n", "batch_size"],
        output_file="strided_speedup.pdf",
        title="Performance Improvement of Patched vs Vanilla Implementation"
    )
    
    print("Visualization complete!")