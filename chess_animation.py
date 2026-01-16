#!/usr/bin/env python3
"""
Chess Game Animation Generator
==============================

Creates animated game replays with progressive plot reveals.
Takes analysis output from chess_game_analyzer and generates GIF/MP4 animations.

Layout:
+------------------+--------------------+
|                  |  Eval plot         |
|                  |  (progressively    |
|    Chess         |   revealed)        |
|    Board         +--------------------+
|                  |  Space plot        |
|    (current      +--------------------+
|     position)    |  Mobility plot     |
|                  +--------------------+
|                  |  Threats plot      |
+------------------+--------------------+

The plots are revealed left-to-right as the game progresses,
with a semi-transparent mask covering future moves.

Usage:
    from chess_game_analyzer6h import analyze_game_with_positional_metrics
    from chess_animation import generate_game_animation
    
    # First, analyze the game
    result = analyze_game_with_positional_metrics("game.pgn")
    
    # Generate animation
    generate_game_animation(
        result,
        output_path="game_replay.gif",
        frame_duration=1.0
    )
    
    # Or generate MP4 (requires ffmpeg)
    generate_game_animation(
        result,
        output_path="game_replay.mp4",
        frame_duration=0.5
    )

Dependencies:
    - python-chess (for board rendering)
    - matplotlib (for plots)
    - Pillow (PIL) (for image manipulation)
    - cairosvg (optional, for better SVG rendering)
    - imageio (optional, for MP4 output)

Author: Generated for David Joyner's chess analysis pipeline
License: Modified BSD or MIT (user's choice)
"""

import chess
import chess.svg
import io
import os
from typing import List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

# Check for required dependencies
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

if TYPE_CHECKING:
    from chess_game_analyzer6h import EnhancedGameAnalysisResult, EnhancedMoveAnalysis


@dataclass
class AnimationConfig:
    """Configuration for animation generation."""
    board_size: int = 400          # Board size in pixels
    plot_width: int = 350          # Width of plot panel in pixels
    plot_height: int = 100         # Height of each individual plot
    frame_duration: float = 1.0    # Seconds per frame
    dpi: int = 100                 # Resolution for plots
    mask_color: Tuple[int, int, int] = (128, 128, 128)  # Gray mask
    mask_alpha: float = 0.7        # Mask transparency (0-1)
    background_color: Tuple[int, int, int] = (255, 255, 255)  # White background
    show_move_text: bool = True    # Show current move annotation
    show_eval_text: bool = True    # Show current evaluation
    font_size: int = 14            # Font size for annotations
    margin: int = 10               # Margin between elements
    include_initial_position: bool = True  # Include starting position as first frame


def check_dependencies() -> dict:
    """Check which dependencies are available."""
    return {
        'PIL': PIL_AVAILABLE,
        'matplotlib': MATPLOTLIB_AVAILABLE,
        'cairosvg': CAIROSVG_AVAILABLE,
        'imageio': IMAGEIO_AVAILABLE
    }


def render_board_to_image(board: chess.Board, size: int = 400, 
                          last_move: chess.Move = None) -> 'Image.Image':
    """
    Render a chess board to a PIL Image.
    
    Args:
        board: Current board state
        size: Size of the board in pixels
        last_move: Optional last move to highlight
        
    Returns:
        PIL Image of the board
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL (Pillow) is required for board rendering")
    
    # Generate SVG
    svg_data = chess.svg.board(
        board, 
        size=size,
        lastmove=last_move,
        colors={
            'square light': '#F0D9B5',
            'square dark': '#B58863',
            'square light lastmove': '#CDD16A',
            'square dark lastmove': '#AAA23B',
        }
    )
    
    # Convert SVG to PNG
    if CAIROSVG_AVAILABLE:
        # Higher quality rendering with cairosvg
        png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'), 
                                     output_width=size, output_height=size)
        return Image.open(io.BytesIO(png_data)).convert('RGB')
    else:
        # Fallback: use a simple text-based representation or require cairosvg
        # For now, create a placeholder with piece positions
        return _render_board_fallback(board, size)


def _render_board_fallback(board: chess.Board, size: int) -> 'Image.Image':
    """
    Fallback board rendering without cairosvg.
    Creates a simple but functional board image.
    """
    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)
    
    square_size = size // 8
    
    # Colors
    light_square = (240, 217, 181)  # #F0D9B5
    dark_square = (181, 136, 99)    # #B58863
    
    # Draw squares
    for rank in range(8):
        for file in range(8):
            x1 = file * square_size
            y1 = (7 - rank) * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size
            
            color = light_square if (rank + file) % 2 == 0 else dark_square
            draw.rectangle([x1, y1, x2, y2], fill=color)
    
    # Draw pieces as text (simple fallback)
    piece_symbols = {
        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
    }
    
    try:
        # Try to use a font that supports chess symbols
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 
                                   int(square_size * 0.7))
    except:
        font = ImageFont.load_default()
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            x = file * square_size + square_size // 4
            y = (7 - rank) * square_size + square_size // 8
            
            symbol = piece_symbols.get(piece.symbol(), piece.symbol())
            color = 'white' if piece.color == chess.WHITE else 'black'
            
            # Draw with outline for visibility
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                draw.text((x + dx, y + dy), symbol, font=font, 
                         fill='black' if color == 'white' else 'white')
            draw.text((x, y), symbol, font=font, fill=color)
    
    return img


def render_plots_with_mask(
    analysis: 'EnhancedGameAnalysisResult',
    reveal_fraction: float,
    config: AnimationConfig
) -> 'Image.Image':
    """
    Render the 4 stacked plots with a semi-transparent mask.
    
    Args:
        analysis: Game analysis result from CGA
        reveal_fraction: Fraction of plot to reveal (0.0 to 1.0)
        config: Animation configuration
        
    Returns:
        PIL Image of the plots with mask applied
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plot rendering")
    if not PIL_AVAILABLE:
        raise ImportError("PIL (Pillow) is required for image manipulation")
    
    moves = analysis.moves
    if not moves:
        # Return blank image if no moves
        return Image.new('RGB', (config.plot_width, config.plot_height * 4), 'white')
    
    # Extract data for all metrics
    plies = [m.ply for m in moves]
    move_numbers = [(p + 1) / 2 for p in plies]
    
    # Evaluation data
    evals = [m.eval_after / 100.0 for m in moves]  # Convert to pawns
    
    # Positional metrics
    space_white = []
    space_black = []
    mobility_white = []
    mobility_black = []
    threats_white = []
    threats_black = []
    
    for m in moves:
        if m.positional_eval:
            pe = m.positional_eval
            space_white.append(pe.space_white)
            space_black.append(pe.space_black)
            mobility_white.append(pe.mobility_white)
            mobility_black.append(pe.mobility_black)
            threats_white.append(pe.threats_white)
            threats_black.append(pe.threats_black)
        else:
            space_white.append(0)
            space_black.append(0)
            mobility_white.append(0)
            mobility_black.append(0)
            threats_white.append(0)
            threats_black.append(0)
    
    # Create figure with 4 subplots
    fig_height = (config.plot_height * 4) / config.dpi
    fig_width = config.plot_width / config.dpi
    
    fig, axes = plt.subplots(4, 1, figsize=(fig_width, fig_height), dpi=config.dpi)
    fig.patch.set_facecolor('white')
    
    # Plot configurations
    plot_configs = [
        {
            'ax': axes[0],
            'white_data': evals,
            'black_data': [-e for e in evals],  # Mirror for black perspective
            'title': 'Evaluation',
            'ylabel': 'Pawns',
            'show_zero_line': True,
            'fill_regions': True
        },
        {
            'ax': axes[1],
            'white_data': space_white,
            'black_data': space_black,
            'title': 'Space',
            'ylabel': 'Score',
            'show_zero_line': False,
            'fill_regions': False
        },
        {
            'ax': axes[2],
            'white_data': mobility_white,
            'black_data': mobility_black,
            'title': 'Mobility',
            'ylabel': 'Score',
            'show_zero_line': False,
            'fill_regions': False
        },
        {
            'ax': axes[3],
            'white_data': threats_white,
            'black_data': threats_black,
            'title': 'Threats',
            'ylabel': 'Score',
            'show_zero_line': False,
            'fill_regions': False
        }
    ]
    
    for pc in plot_configs:
        ax = pc['ax']
        
        # Plot lines
        ax.plot(move_numbers, pc['white_data'], 'b-', linewidth=1.2, label='White')
        ax.plot(move_numbers, pc['black_data'], 'r--', linewidth=1.2, label='Black')
        
        # Zero line for eval
        if pc['show_zero_line']:
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        
        # Fill regions for eval
        if pc['fill_regions']:
            ax.fill_between(move_numbers, 0, pc['white_data'],
                           where=[v > 0 for v in pc['white_data']],
                           alpha=0.2, color='blue')
            ax.fill_between(move_numbers, 0, pc['white_data'],
                           where=[v < 0 for v in pc['white_data']],
                           alpha=0.2, color='red')
        
        # Labels
        ax.set_ylabel(pc['ylabel'], fontsize=8)
        ax.set_title(pc['title'], fontsize=9, fontweight='bold')
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(True, alpha=0.3)
        
        # Only show x-label on bottom plot
        if ax != axes[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Move', fontsize=8)
        
        # Add legend only to first plot
        if ax == axes[0]:
            ax.legend(loc='upper right', fontsize=7)
        
        # Apply mask overlay
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Calculate mask position
        total_width = xlim[1] - xlim[0]
        reveal_x = xlim[0] + total_width * reveal_fraction
        
        # Draw semi-transparent gray rectangle over unrevealed portion
        if reveal_fraction < 1.0:
            rect = patches.Rectangle(
                (reveal_x, ylim[0]),
                xlim[1] - reveal_x,
                ylim[1] - ylim[0],
                facecolor='gray',
                alpha=config.mask_alpha,
                zorder=10  # Draw on top
            )
            ax.add_patch(rect)
        
        # Add vertical line at current position
        if reveal_fraction > 0:
            ax.axvline(x=reveal_x, color='green', linestyle='-', 
                      linewidth=1.5, alpha=0.8, zorder=11)
    
    plt.tight_layout(pad=0.5)
    
    # Convert figure to PIL Image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # Get the RGBA buffer
    buf = canvas.buffer_rgba()
    img = Image.frombuffer('RGBA', canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)
    img = img.convert('RGB')
    
    plt.close(fig)
    
    return img


def create_frame(
    board: chess.Board,
    analysis: 'EnhancedGameAnalysisResult',
    move_index: int,
    last_move: chess.Move,
    config: AnimationConfig
) -> 'Image.Image':
    """
    Create a single animation frame.
    
    Args:
        board: Current board state
        analysis: Game analysis result
        move_index: Current move index (-1 for initial position)
        last_move: The move that was just played (None for initial)
        config: Animation configuration
        
    Returns:
        PIL Image of the complete frame
    """
    moves = analysis.moves
    N = len(moves)
    
    # Calculate reveal fraction
    if move_index < 0:
        reveal_fraction = 0.0
    else:
        reveal_fraction = (move_index + 1) / N
    
    # Render board
    board_img = render_board_to_image(board, config.board_size, last_move)
    
    # Render plots with mask
    plots_img = render_plots_with_mask(analysis, reveal_fraction, config)
    
    # Calculate total dimensions
    total_width = config.board_size + config.margin + config.plot_width
    total_height = max(config.board_size, config.plot_height * 4)
    
    # Add space for text annotations
    text_height = 60 if (config.show_move_text or config.show_eval_text) else 0
    total_height += text_height
    
    # Create combined image
    combined = Image.new('RGB', (total_width, total_height), config.background_color)
    
    # Paste board (vertically centered if plots are taller)
    board_y = (total_height - text_height - config.board_size) // 2
    combined.paste(board_img, (0, board_y))
    
    # Paste plots
    plots_y = (total_height - text_height - plots_img.height) // 2
    combined.paste(plots_img, (config.board_size + config.margin, plots_y))
    
    # Add text annotations
    if config.show_move_text or config.show_eval_text:
        draw = ImageDraw.Draw(combined)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 
                                       config.font_size)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
                                            config.font_size)
        except:
            font = ImageFont.load_default()
            font_bold = font
        
        text_y = total_height - text_height + 10
        
        if move_index >= 0 and move_index < N:
            move_analysis = moves[move_index]
            
            if config.show_move_text:
                move_num = (move_analysis.ply + 1) // 2
                move_color = "White" if move_analysis.is_white_move else "Black"
                move_text = f"Move {move_num}: {move_analysis.move_san} ({move_color})"
                draw.text((10, text_y), move_text, font=font_bold, fill='black')
            
            if config.show_eval_text:
                eval_pawns = move_analysis.eval_after / 100.0
                if eval_pawns > 0:
                    eval_text = f"Eval: +{eval_pawns:.2f} (White)"
                    eval_color = 'blue'
                else:
                    eval_text = f"Eval: {eval_pawns:.2f} (Black)"
                    eval_color = 'red'
                draw.text((10, text_y + 22), eval_text, font=font, fill=eval_color)
        else:
            # Initial position
            draw.text((10, text_y), "Starting Position", font=font_bold, fill='black')
            draw.text((10, text_y + 22), "Eval: 0.00 (Equal)", font=font, fill='gray')
    
    return combined


def generate_game_animation(
    analysis: 'EnhancedGameAnalysisResult',
    output_path: str,
    config: AnimationConfig = None,
    frame_duration: float = None,
    progress_callback=None
) -> bool:
    """
    Generate an animated game replay with progressive plot reveals.
    
    Args:
        analysis: EnhancedGameAnalysisResult from chess_game_analyzer
        output_path: Output file path (.gif, .mp4, or .png for frame sequence)
        config: AnimationConfig object (uses defaults if None)
        frame_duration: Override config.frame_duration if provided
        progress_callback: Optional callback(current, total) for progress updates
        
    Returns:
        True if successful, False otherwise
    """
    # Check dependencies
    if not PIL_AVAILABLE:
        print("Error: PIL (Pillow) is required. Install with: pip install Pillow")
        return False
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required. Install with: pip install matplotlib")
        return False
    
    # Setup config
    if config is None:
        config = AnimationConfig()
    if frame_duration is not None:
        config.frame_duration = frame_duration
    
    moves = analysis.moves
    N = len(moves)
    
    if N == 0:
        print("Error: No moves in the game to animate")
        return False
    
    print(f"Generating animation for {N} moves...")
    print(f"Output: {output_path}")
    
    # Determine output format
    output_ext = os.path.splitext(output_path)[1].lower()
    
    frames = []
    board = chess.Board()
    
    # Generate frames
    total_frames = N + (1 if config.include_initial_position else 0)
    
    # Initial position frame
    if config.include_initial_position:
        if progress_callback:
            progress_callback(0, total_frames)
        frame = create_frame(board, analysis, -1, None, config)
        frames.append(frame)
        print(f"  Frame 0/{total_frames}: Initial position")
    
    # Move frames
    for i, move_analysis in enumerate(moves):
        if progress_callback:
            progress_callback(i + 1, total_frames)
        
        # Parse and make the move
        move = chess.Move.from_uci(move_analysis.move_uci)
        board.push(move)
        
        # Create frame
        frame = create_frame(board, analysis, i, move, config)
        frames.append(frame)
        
        if (i + 1) % 10 == 0 or i == N - 1:
            print(f"  Frame {i + 1}/{total_frames}: {move_analysis.move_san}")
    
    # Save animation
    print(f"Saving animation ({len(frames)} frames)...")
    
    if output_ext == '.gif':
        # Save as GIF
        duration_ms = int(config.frame_duration * 1000)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0  # Loop forever
        )
        print(f"Saved GIF: {output_path}")
        return True
    
    elif output_ext == '.mp4':
        # Save as MP4 (requires imageio)
        if not IMAGEIO_AVAILABLE:
            print("Error: imageio is required for MP4 output. Install with: pip install imageio[ffmpeg]")
            return False
        
        # Convert PIL images to numpy arrays
        import numpy as np
        frames_np = [np.array(f) for f in frames]
        
        fps = 1.0 / config.frame_duration
        imageio.mimwrite(output_path, frames_np, fps=fps)
        print(f"Saved MP4: {output_path}")
        return True
    
    elif output_ext == '.png':
        # Save as PNG sequence
        base_path = output_path.replace('.png', '')
        for i, frame in enumerate(frames):
            frame_path = f"{base_path}_{i:04d}.png"
            frame.save(frame_path)
        print(f"Saved {len(frames)} PNG frames: {base_path}_XXXX.png")
        return True
    
    else:
        print(f"Error: Unsupported output format '{output_ext}'. Use .gif, .mp4, or .png")
        return False


def generate_single_frame(
    analysis: 'EnhancedGameAnalysisResult',
    move_index: int,
    output_path: str,
    config: AnimationConfig = None
) -> bool:
    """
    Generate a single frame at a specific move.
    Useful for creating thumbnails or specific position snapshots.
    
    Args:
        analysis: EnhancedGameAnalysisResult from chess_game_analyzer
        move_index: Move index (0-based, -1 for initial position)
        output_path: Output PNG file path
        config: AnimationConfig object (uses defaults if None)
        
    Returns:
        True if successful, False otherwise
    """
    if not PIL_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        print("Error: PIL and matplotlib are required")
        return False
    
    if config is None:
        config = AnimationConfig()
    
    moves = analysis.moves
    board = chess.Board()
    last_move = None
    
    # Replay to the specified position
    for i in range(move_index + 1):
        if i >= 0 and i < len(moves):
            move = chess.Move.from_uci(moves[i].move_uci)
            board.push(move)
            last_move = move
    
    # Create and save frame
    frame = create_frame(board, analysis, move_index, last_move, config)
    frame.save(output_path)
    print(f"Saved frame: {output_path}")
    return True


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Command line interface for chess animation generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate animated chess game replays with progressive plot reveals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate GIF animation (requires analyzed game JSON)
  python chess_animation.py analysis.json -o game_replay.gif
  
  # Generate MP4 with custom frame duration
  python chess_animation.py analysis.json -o game_replay.mp4 --fps 2
  
  # Generate PNG sequence
  python chess_animation.py analysis.json -o frames.png
  
  # Custom board size
  python chess_animation.py analysis.json -o game.gif --board-size 500

Note: This tool requires a JSON analysis file from chess_game_analyzer.
      Generate one with: python chess_game_analyzer6h.py game.pgn --json-output analysis.json
        """
    )
    
    parser.add_argument("analysis_json", help="Path to analysis JSON file from chess_game_analyzer")
    parser.add_argument("-o", "--output", required=True, help="Output file (.gif, .mp4, or .png)")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second (default: 1.0)")
    parser.add_argument("--board-size", type=int, default=400, help="Board size in pixels (default: 400)")
    parser.add_argument("--plot-width", type=int, default=350, help="Plot panel width (default: 350)")
    parser.add_argument("--no-text", action="store_true", help="Don't show move/eval text annotations")
    parser.add_argument("--no-initial", action="store_true", help="Don't include initial position frame")
    
    args = parser.parse_args()
    
    # Check dependencies
    deps = check_dependencies()
    missing = [k for k, v in deps.items() if not v and k in ['PIL', 'matplotlib']]
    if missing:
        print(f"Error: Missing required dependencies: {', '.join(missing)}")
        print("Install with: pip install Pillow matplotlib")
        return
    
    # Load analysis JSON
    import json
    from dataclasses import dataclass, field
    from typing import List, Dict, Optional
    
    print(f"Loading analysis from {args.analysis_json}...")
    
    with open(args.analysis_json, 'r') as f:
        data = json.load(f)
    
    # Reconstruct minimal analysis object from JSON
    # This is a simplified reconstruction - full reconstruction would need
    # the actual dataclass definitions
    
    @dataclass
    class SimplePositionalEval:
        space_white: float = 0.0
        space_black: float = 0.0
        mobility_white: float = 0.0
        mobility_black: float = 0.0
        threats_white: float = 0.0
        threats_black: float = 0.0
        king_safety_white: float = 0.0
        king_safety_black: float = 0.0
    
    @dataclass
    class SimpleMoveAnalysis:
        ply: int
        move_san: str
        move_uci: str
        is_white_move: bool
        eval_after: float
        positional_eval: Optional[SimplePositionalEval] = None
    
    @dataclass
    class SimpleAnalysisResult:
        moves: List[SimpleMoveAnalysis] = field(default_factory=list)
    
    # Parse moves from JSON
    analysis = SimpleAnalysisResult()
    for move_data in data.get('moves', []):
        pe_data = move_data.get('positional_eval')
        pe = None
        if pe_data:
            pe = SimplePositionalEval(
                space_white=pe_data.get('space_white', 0),
                space_black=pe_data.get('space_black', 0),
                mobility_white=pe_data.get('mobility_white', 0) or pe_data.get('mobility_white_mg', 0),
                mobility_black=pe_data.get('mobility_black', 0) or pe_data.get('mobility_black_mg', 0),
                threats_white=pe_data.get('threats_white', 0) or pe_data.get('threats_white_mg', 0),
                threats_black=pe_data.get('threats_black', 0) or pe_data.get('threats_black_mg', 0),
            )
        
        move = SimpleMoveAnalysis(
            ply=move_data['ply'],
            move_san=move_data['move_san'],
            move_uci=move_data['move_uci'],
            is_white_move=move_data['is_white_move'],
            eval_after=move_data['eval_after'],
            positional_eval=pe
        )
        analysis.moves.append(move)
    
    print(f"Loaded {len(analysis.moves)} moves")
    
    # Configure animation
    config = AnimationConfig(
        board_size=args.board_size,
        plot_width=args.plot_width,
        frame_duration=1.0 / args.fps,
        show_move_text=not args.no_text,
        show_eval_text=not args.no_text,
        include_initial_position=not args.no_initial
    )
    
    # Generate animation
    success = generate_game_animation(analysis, args.output, config)
    
    if success:
        print("Animation generation complete!")
    else:
        print("Animation generation failed.")


if __name__ == "__main__":
    main()
