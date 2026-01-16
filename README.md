# Chess Game Analyzer Suite

A comprehensive Python toolkit for deep analysis of chess games, generating professional LaTeX reports with positional metrics, evaluation graphs, game character classification, and animated game replays.

## Modules

| Module | Purpose |
|--------|---------|
| `chess_game_analyzer.py` | Core analysis engine — Stockfish integration, positional metrics, LaTeX reports |
| `chess_plotting.py` | Plotting utilities — matplotlib and ASCII graphs for metrics over time |
| `chess_animation.py` | Animation generator — GIF/MP4 replays with progressive plot reveals |

## Features

- **Stockfish Integration**: Move-by-move centipawn evaluation with configurable depth
- **Positional Metrics**: Space control, piece mobility, king safety, and threats
- **Game Character Classification**: Automatic categorization (balanced/tense/tactical/chaotic)
- **Multi-PV Analysis**: Suggests playable alternative moves within 50cp of best
- **Brilliant Sacrifice Detection**: Identifies and highlights spectacular sacrifices
- **Critical Position Identification**: Marks turning points and biggest evaluation swings
- **LaTeX Report Generation**: Publication-ready reports with chess diagrams and plots
- **Multi-Game Book Support**: Analyze entire PGN databases and generate book-format documents
- **Animated Replays**: GIF/MP4 animations with progressive plot reveals

---

## Installation

### Requirements

- Python 3.8+
- Stockfish chess engine

### Python Dependencies

```bash
# Core dependencies
pip install python-chess matplotlib

# For animations
pip install Pillow cairosvg

# For MP4 output (optional)
pip install imageio[ffmpeg]
```

### Installing Stockfish

**macOS:**
```bash
brew install stockfish
# Usually installs to /usr/local/bin/stockfish or /opt/homebrew/bin/stockfish
```

**Ubuntu/Debian:**
```bash
sudo apt install stockfish
# Installs to /usr/games/stockfish
```

**Windows:**
Download from [stockfishchess.org](https://stockfishchess.org/download/)

### LaTeX Requirements (for PDF generation)

The generated `.tex` files require these LaTeX packages:
- `xskak` and `chessboard` — chess diagrams
- `pgfplots` — evaluation graphs
- `booktabs`, `longtable`, `hyperref`, `xcolor`, `graphicx`

---

## Module 1: chess_game_analyzer.py

The core analysis engine that processes PGN files and generates detailed analysis.

### Quick Start

```python
from chess_game_analyzer6h import analyze_game_with_positional_metrics

# Analyze a single game
result = analyze_game_with_positional_metrics(
    pgn_source="game.pgn",
    stockfish_path="/usr/local/bin/stockfish",  # Adjust for your system
    output_path="analysis.tex",
    include_plots=True,
    plot_output_dir="./plots/"
)

# Access results
print(f"White: {result.white} ({result.white_elo})")
print(f"Black: {result.black} ({result.black_elo})")
print(f"Result: {result.result}")
print(f"White accuracy: {result.white_stats['accuracy']:.1f}%")
print(f"Black accuracy: {result.black_stats['accuracy']:.1f}%")
```

### Game Character Classification

```python
# Access game character classification
gc = result.game_character
print(f"Game type: {gc['spread_class']} ({gc['direction_class']})")
print(f"Evaluation range: {gc['m1']:+.2f} to {gc['m2']:+.2f}")
print(f"Spread: {gc['spread']:.2f}")
print(f"Description: {gc['combined_description']}")
```

Classification ranges:
| Spread (d) | Class | Description |
|------------|-------|-------------|
| d < 1 | **Balanced** | Minimal advantage shifts |
| 1 ≤ d < 3 | **Tense** | Normal competitive tension |
| 3 ≤ d < 6 | **Tactical** | Significant swings |
| d ≥ 6 | **Chaotic** | Wild swings, likely blunders |

### Analyzing Move Quality

```python
# Iterate through moves
for move in result.moves:
    print(f"Move {move.ply}: {move.move_san}")
    print(f"  Classification: {move.classification}")
    print(f"  Eval: {move.eval_after/100:+.2f}")
    print(f"  Best was: {move.best_move_san}")
    
    # Alternative moves (within 50cp of best)
    if move.alternative_moves:
        alts = ", ".join(san for san, _ in move.alternative_moves)
        print(f"  Also playable: {alts}")
    
    # Positional metrics
    if move.positional_eval:
        pe = move.positional_eval
        print(f"  Space: W={pe.space_white:.1f} B={pe.space_black:.1f}")
        print(f"  Mobility: W={pe.mobility_white:.1f} B={pe.mobility_black:.1f}")
```

### Multi-Game Book Generation

```python
from chess_game_analyzer6h import analyze_games_to_book

# Analyze all games in a PGN file and generate a book
analyze_games_to_book(
    pgn_source="tournament.pgn",
    output_path="tournament_analysis.tex",
    stockfish_path="/usr/local/bin/stockfish",
    book_title="2024 Club Championship",
    book_author="Analysis by Stockfish 16",
    include_plots=True,
    plot_output_dir="./plots/"
)
```

### Command Line Usage

```bash
# Single game analysis
python chess_game_analyzer.py game.pgn -o analysis.tex -s /usr/local/bin/stockfish

# Multi-game book
python chess_game_analyzer.py games.pgn -o book.tex --book --book-title "My Games"

# JSON output for further processing
python chess_game_analyzer.py game.pgn --json-output analysis.json

# Custom depth and time
python chess_game_analyzer.py game.pgn -o analysis.tex -d 24 -t 2.0

# Skip plots and methodology
python chess_game_analyzer.py game.pgn -o analysis.tex --no-plots --no-methodology
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output LaTeX file | None |
| `--json-output` | Output JSON file | None |
| `-s, --stockfish` | Path to Stockfish | `/usr/games/stockfish` |
| `-d, --depth` | Analysis depth | 20 |
| `-t, --time` | Seconds per position | 1.0 |
| `--no-diagrams` | Skip board diagrams | False |
| `--no-methodology` | Skip methodology section | False |
| `--no-plots` | Skip matplotlib plots | False |
| `--ascii-plots` | Include ASCII plots | False |
| `--book` | Generate book format | False |
| `--book-title` | Book title | "Chess Game Collection Analysis" |
| `--book-author` | Book author | Engine version |
| `-q, --quiet` | Suppress progress | False |

---

## Module 2: chess_plotting.py

Generates plots of chess game metrics for LaTeX reports or standalone use.

### Matplotlib Plots

```python
from chess_plotting import ChessPlotter

# Generate evaluation plot
ChessPlotter.generate_matplotlib_plot(
    result.moves, 
    metric='eval', 
    output_path='plot_eval.pdf',
    title='Position Evaluation',
    figsize=(10, 4)
)

# Generate space control plot
ChessPlotter.generate_matplotlib_plot(
    result.moves,
    metric='space',
    output_path='plot_space.pdf'
)

# Available metrics: 'eval', 'space', 'mobility', 'king_safety', 'threats'
```

### ASCII Plots (for LaTeX verbatim)

```python
from chess_plotting import ChessPlotter

# Generate ASCII art plot
ascii_plot = ChessPlotter.generate_ascii_plot(
    result.moves,
    metric='eval',
    width=70,
    height=20
)
print(ascii_plot)
```

Output:
```
                        Position Evaluation (pawns)

  2.50 |                                              W
       |                                         W
  1.25 |                              W      W
       |                         W
  0.00 |....W...W...W...W...W...............................
       |         B   B   B   B   B
 -1.25 |                              B      B
       |                                         B
 -2.50 |                                              B
       +------------------------------------------------
         1        10        20        30        40
                          Move Number

Legend: W = White   B = Black   X = overlap
```

### Check Matplotlib Availability

```python
from chess_plotting import is_matplotlib_available

if is_matplotlib_available():
    print("Matplotlib plots available")
else:
    print("Falling back to ASCII plots")
```

---

## Module 3: chess_animation.py

Creates animated game replays with progressive plot reveals.

### Basic Animation

```python
from chess_game_analyzer6h import analyze_game_with_positional_metrics
from chess_animation import generate_game_animation

# First analyze the game
result = analyze_game_with_positional_metrics(
    pgn_source="game.pgn",
    stockfish_path="/usr/local/bin/stockfish"
)

# Generate GIF animation
generate_game_animation(
    result,
    output_path="game_replay.gif",
    frame_duration=1.0  # 1 second per move
)
```

### Custom Configuration

```python
from chess_animation import generate_game_animation, AnimationConfig

# Create custom configuration
config = AnimationConfig(
    board_size=500,           # Larger board
    plot_width=400,           # Wider plots
    plot_height=100,          # Height per plot
    frame_duration=0.5,       # Faster playback (2 FPS)
    mask_alpha=0.6,           # Lighter mask
    show_move_text=True,      # Show move annotations
    show_eval_text=True,      # Show evaluation
    include_initial_position=True  # Start with empty board
)

generate_game_animation(result, "game_replay.gif", config)
```

### Output Formats

```python
# GIF (default, widely compatible)
generate_game_animation(result, "replay.gif")

# MP4 (smaller file size, requires imageio[ffmpeg])
generate_game_animation(result, "replay.mp4")

# PNG sequence (for custom video processing)
generate_game_animation(result, "frames.png")
# Creates: frames_0000.png, frames_0001.png, ...
```

### Single Frame Export

```python
from chess_animation import generate_single_frame, AnimationConfig

# Export a specific position
generate_single_frame(
    result,
    move_index=25,  # After move 25 (0-based)
    output_path="position_25.png"
)

# Export initial position
generate_single_frame(result, move_index=-1, output_path="start.png")
```

### Command Line Usage

```bash
# First generate JSON analysis
python chess_game_analyzer.py game.pgn --json-output analysis.json -s /usr/local/bin/stockfish

# Generate GIF from JSON
python chess_animation.py analysis.json -o replay.gif

# Custom options
python chess_animation.py analysis.json -o replay.gif --fps 2 --board-size 500

# Generate MP4
python chess_animation.py analysis.json -o replay.mp4 --fps 3
```

### Animation Layout

```
+------------------+--------------------+
|                  |  Eval plot         |
|    Chess         |  ████████░░░░░░░░  |  <- Gray mask reveals progressively
|    Board         +--------------------+
|                  |  Space plot        |
|   (current       |  ████████░░░░░░░░  |
|    position)     +--------------------+
|                  |  Mobility plot     |
|    ♜ ♞ ♝ ♛ ♚     |  ████████░░░░░░░░  |
|    ♟ ♟ ♟ ♟ ♟     +--------------------+
|                  |  Threats plot      |
|                  |  ████████░░░░░░░░  |
+------------------+--------------------+
|  Move 15: Nf3 (White)    Eval: +1.25  |
+---------------------------------------+
```

### Dependencies Check

```python
from chess_animation import check_dependencies

deps = check_dependencies()
print(f"PIL (Pillow): {deps['PIL']}")
print(f"matplotlib: {deps['matplotlib']}")
print(f"cairosvg: {deps['cairosvg']}")      # Recommended for board rendering
print(f"imageio: {deps['imageio']}")        # Required for MP4
```

---

## Complete Workflow Example

```python
from chess_game_analyzer6h import analyze_game_with_positional_metrics
from chess_animation import generate_game_animation, AnimationConfig

# 1. Analyze the game
print("Analyzing game...")
result = analyze_game_with_positional_metrics(
    pgn_source="kasparov_deep_blue_1997_game6.pgn",
    stockfish_path="/usr/local/bin/stockfish",
    output_path="analysis.tex",
    include_plots=True,
    plot_output_dir="./plots/",
    depth=22,
    time_limit=2.0
)

# 2. Print summary
print(f"\n{result.white} vs {result.black}")
print(f"Result: {result.result}")
print(f"Opening: {result.opening_eco} - {result.opening_name}")
print(f"\nGame Character: {result.game_character['combined_description']}")
print(f"White accuracy: {result.white_stats['accuracy']:.1f}%")
print(f"Black accuracy: {result.black_stats['accuracy']:.1f}%")

# 3. Find critical moments
print("\nCritical positions:")
for pos in result.critical_positions:
    move_num = (pos.ply + 1) // 2
    print(f"  Move {move_num}: {pos.move_san} - {pos.reason}")

# 4. List brilliant sacrifices
if result.brilliant_sacrifices:
    print("\nBrilliant sacrifices:")
    for sac in result.brilliant_sacrifices:
        print(f"  {sac.player}: {sac.move_san} ({sac.piece_type})")

# 5. Generate animation
print("\nGenerating animation...")
config = AnimationConfig(
    board_size=450,
    frame_duration=0.75
)
generate_game_animation(result, "game_replay.gif", config)

print("\nDone! Files created:")
print("  - analysis.tex (LaTeX report)")
print("  - plots/*.pdf (metric plots)")
print("  - game_replay.gif (animated replay)")
```

---

## Data Classes Reference

### EnhancedGameAnalysisResult

| Field | Type | Description |
|-------|------|-------------|
| `white`, `black` | str | Player names |
| `white_elo`, `black_elo` | int | Player ratings |
| `result` | str | Game result ("1-0", "0-1", "1/2-1/2") |
| `date`, `event`, `site` | str | Game metadata |
| `opening_eco`, `opening_name` | str | Opening classification |
| `moves` | List[EnhancedMoveAnalysis] | Move-by-move analysis |
| `brilliant_sacrifices` | List[BrilliantSacrifice] | Detected sacrifices |
| `critical_positions` | List[CriticalPosition] | Key moments |
| `white_stats`, `black_stats` | Dict | Player statistics |
| `game_character` | Dict | Game classification |
| `positional_summary` | Dict | Aggregated metrics |

### EnhancedMoveAnalysis

| Field | Type | Description |
|-------|------|-------------|
| `ply` | int | Half-move number |
| `move_san`, `move_uci` | str | Move in SAN/UCI format |
| `is_white_move` | bool | True if White's move |
| `eval_before`, `eval_after` | float | Centipawn evaluation |
| `best_move_san`, `best_move_uci` | str | Engine's best move |
| `eval_loss` | float | Centipawns lost |
| `classification` | str | "best", "excellent", "good", "inaccuracy", "mistake", "blunder" |
| `alternative_moves` | List[Tuple[str, float]] | Playable alternatives (SAN, eval) |
| `positional_eval` | PositionalEvaluation | Detailed positional metrics |

### PositionalEvaluation

| Field | Type | Description |
|-------|------|-------------|
| `space_white`, `space_black` | float | Space control score |
| `mobility_white`, `mobility_black` | float | Piece mobility score |
| `king_safety_white`, `king_safety_black` | float | King safety score |
| `threats_white`, `threats_black` | float | Threat score |

---

## License

Modified BSD or MIT License (user's choice)

## Author

Generated for David Joyner's chess analysis pipeline, 2026

## Contributing

Contributions welcome! Please ensure changes maintain compatibility with the LaTeX output format and include appropriate documentation.
