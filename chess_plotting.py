#!/usr/bin/env python3
"""
Chess Plotting Utilities
========================

Generates plots of chess game metrics for LaTeX reports.
Supports both matplotlib (PDF/PNG) and ASCII (for LaTeX verbatim).

Usage:
    from chess_plotting import ChessPlotter
    
    # Generate matplotlib plot
    ChessPlotter.generate_matplotlib_plot(moves, 'eval', 'plot_eval.pdf')
    
    # Generate ASCII plot
    ascii_plot = ChessPlotter.generate_ascii_plot(moves, 'space')
"""

from typing import List, Tuple, TYPE_CHECKING

# Optional matplotlib import
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server use
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

if TYPE_CHECKING:
    from chess_game_analyzer5 import EnhancedMoveAnalysis


class ChessPlotter:
    """
    Generates plots of chess game metrics.
    Supports both matplotlib (PDF/PNG) and ASCII (for LaTeX verbatim).
    """
    
    @staticmethod
    def generate_matplotlib_plot(
        moves: List['EnhancedMoveAnalysis'],
        metric: str,
        output_path: str,
        title: str = None,
        ylabel: str = None,
        figsize: Tuple[int, int] = (10, 4)
    ) -> bool:
        """
        Generate a matplotlib plot of a metric over the game.
        
        Args:
            moves: List of move analyses (EnhancedMoveAnalysis objects)
            metric: One of 'eval', 'space', 'mobility', 'king_safety'
            output_path: Path to save the plot (PDF or PNG)
            title: Plot title (auto-generated if None)
            ylabel: Y-axis label (auto-generated if None)
            figsize: Figure size tuple
            
        Returns:
            True if successful, False if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            return False
        
        # Extract data based on metric
        plies = []
        white_values = []
        black_values = []
        
        for m in moves:
            plies.append(m.ply)
            
            if metric == 'eval':
                # For eval, we show the position evaluation (positive = White advantage)
                white_values.append(m.eval_after / 100.0)  # Convert to pawns
                black_values.append(-m.eval_after / 100.0)  # Mirror for Black's perspective
            elif m.positional_eval:
                pe = m.positional_eval
                if metric == 'space':
                    white_values.append(pe.space_white)
                    black_values.append(pe.space_black)
                elif metric == 'mobility':
                    white_values.append(pe.mobility_white)
                    black_values.append(pe.mobility_black)
                elif metric == 'king_safety':
                    white_values.append(pe.king_safety_white)
                    black_values.append(pe.king_safety_black)
                elif metric == 'threats':
                    white_values.append(pe.threats_white)
                    black_values.append(pe.threats_black)
                else:
                    white_values.append(0)
                    black_values.append(0)
            else:
                white_values.append(0)
                black_values.append(0)
        
        # Convert ply to move numbers for x-axis
        move_numbers = [(p + 1) / 2 for p in plies]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot White (solid blue) and Black (dotted red)
        ax.plot(move_numbers, white_values, 'b-', linewidth=1.5, label='White')
        ax.plot(move_numbers, black_values, 'r--', linewidth=1.5, label='Black')
        
        # For eval, add a zero line and shade regions
        if metric == 'eval':
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax.fill_between(move_numbers, 0, white_values, 
                           where=[v > 0 for v in white_values], 
                           alpha=0.2, color='blue')
            ax.fill_between(move_numbers, 0, white_values,
                           where=[v < 0 for v in white_values],
                           alpha=0.2, color='red')
        
        # Labels and title
        default_titles = {
            'eval': 'Position Evaluation',
            'space': 'Space Control',
            'mobility': 'Piece Mobility',
            'king_safety': 'King Safety',
            'threats': 'Threats'
        }
        default_ylabels = {
            'eval': 'Evaluation (pawns)',
            'space': 'Space score',
            'mobility': 'Mobility score',
            'king_safety': 'King safety score',
            'threats': 'Threat score'
        }
        
        ax.set_title(title or default_titles.get(metric, metric.title()))
        ax.set_xlabel('Move Number')
        ax.set_ylabel(ylabel or default_ylabels.get(metric, 'Value'))
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Tight layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return True
    
    @staticmethod
    def generate_ascii_plot(
        moves: List['EnhancedMoveAnalysis'],
        metric: str,
        width: int = 70,
        height: int = 20,
        title: str = None
    ) -> str:
        """
        Generate an ASCII art plot suitable for LaTeX verbatim environment.
        
        Args:
            moves: List of move analyses (EnhancedMoveAnalysis objects)
            metric: One of 'eval', 'space', 'mobility', 'king_safety'
            width: Character width of the plot area
            height: Character height of the plot area
            title: Optional title
            
        Returns:
            ASCII string representation of the plot
        """
        # Extract data
        white_values = []
        black_values = []
        
        for m in moves:
            if metric == 'eval':
                white_values.append(m.eval_after / 100.0)
                black_values.append(-m.eval_after / 100.0)  # Inverted for visualization
            elif m.positional_eval:
                pe = m.positional_eval
                if metric == 'space':
                    white_values.append(pe.space_white)
                    black_values.append(pe.space_black)
                elif metric == 'mobility':
                    white_values.append(pe.mobility_white)
                    black_values.append(pe.mobility_black)
                elif metric == 'king_safety':
                    white_values.append(pe.king_safety_white)
                    black_values.append(pe.king_safety_black)
                elif metric == 'threats':
                    white_values.append(pe.threats_white)
                    black_values.append(pe.threats_black)
                else:
                    white_values.append(0)
                    black_values.append(0)
            else:
                white_values.append(0)
                black_values.append(0)
        
        if not white_values:
            return "No data to plot"
        
        # Determine scale
        all_values = white_values + black_values
        min_val = min(all_values)
        max_val = max(all_values)
        
        # Add padding
        val_range = max_val - min_val
        if val_range == 0:
            val_range = 1
            min_val -= 0.5
            max_val += 0.5
        else:
            min_val -= val_range * 0.1
            max_val += val_range * 0.1
        
        # Create canvas
        canvas = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Y-axis labels column width
        y_label_width = 8
        plot_width = width - y_label_width - 2
        
        # Helper to convert value to row
        def val_to_row(v):
            normalized = (v - min_val) / (max_val - min_val)
            row = int((1 - normalized) * (height - 1))
            return max(0, min(height - 1, row))
        
        # Helper to convert index to column
        def idx_to_col(i):
            if len(white_values) == 1:
                return y_label_width + plot_width // 2
            col = y_label_width + int(i * (plot_width - 1) / (len(white_values) - 1))
            return max(y_label_width, min(width - 2, col))
        
        # Draw axes
        for row in range(height):
            canvas[row][y_label_width - 1] = '|'
        for col in range(y_label_width - 1, width):
            canvas[height - 1][col] = '-'
        canvas[height - 1][y_label_width - 1] = '+'
        
        # Draw Y-axis labels
        for i, frac in enumerate([1.0, 0.75, 0.5, 0.25, 0.0]):
            row = int(frac * (height - 2))
            val = max_val - frac * (max_val - min_val)
            label = f"{val:6.2f}"
            for j, c in enumerate(label):
                if j < y_label_width - 2:
                    canvas[row][j] = c
        
        # Draw zero line if in range
        if min_val < 0 < max_val:
            zero_row = val_to_row(0)
            for col in range(y_label_width, width - 1):
                if canvas[zero_row][col] == ' ':
                    canvas[zero_row][col] = '.'
        
        # Plot White values (using 'W')
        for i, v in enumerate(white_values):
            col = idx_to_col(i)
            row = val_to_row(v)
            canvas[row][col] = 'W'
        
        # Plot Black values (using 'B')
        for i, v in enumerate(black_values):
            col = idx_to_col(i)
            row = val_to_row(v)
            if canvas[row][col] == 'W':
                canvas[row][col] = 'X'  # Overlap marker
            else:
                canvas[row][col] = 'B'
        
        # X-axis labels (move numbers)
        x_labels_row = []
        for i in [0, len(moves) // 4, len(moves) // 2, 3 * len(moves) // 4, len(moves) - 1]:
            if i < len(moves):
                move_num = (moves[i].ply + 1) // 2
                col = idx_to_col(i)
                x_labels_row.append((col, str(move_num)))
        
        # Build output
        lines = []
        
        # Title
        default_titles = {
            'eval': 'Position Evaluation (pawns)',
            'space': 'Space Control',
            'mobility': 'Piece Mobility',
            'king_safety': 'King Safety',
            'threats': 'Threats'
        }
        plot_title = title or default_titles.get(metric, metric.title())
        lines.append(plot_title.center(width))
        lines.append('')
        
        # Canvas
        for row in canvas:
            lines.append(''.join(row))
        
        # X-axis labels
        x_label_line = [' '] * width
        for col, label in x_labels_row:
            for j, c in enumerate(label):
                if col + j < width:
                    x_label_line[col + j] = c
        lines.append(''.join(x_label_line))
        lines.append(' ' * (y_label_width + plot_width // 2 - 5) + 'Move Number')
        
        # Legend
        lines.append('')
        lines.append('Legend: W = White   B = Black   X = overlap')
        
        return '\n'.join(lines)


def is_matplotlib_available() -> bool:
    """Check if matplotlib is available for plotting."""
    return MATPLOTLIB_AVAILABLE
