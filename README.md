# Chess Game Animator

A Python pipeline that turns a PGN chess game into an annotated video using [Manim](https://www.manim.community/). Each move is animated on a live board alongside a Stockfish evaluation bar, a scrolling move list, a commentary panel, and a four-plot metrics strip showing positional trends across the whole game.

---

## Example Output

The 16:9 frame is divided into three zones:

```
┌─────────────────────────────────────────────────────────┐
│  Eval │                    │  Header (players, event)   │
│  Bar  │   Chess Board      │  Move List                 │
│       │                    │  Commentary / Analysis     │
├─────────────────────────────────────────────────────────┤
│   Eval [-4, 4]  │  Space  │  Mobility  │  King Safety  │
└─────────────────────────────────────────────────────────┘
```

- **Eval bar** — white/black fill animated each move, scaled to ±4 pawns.
- **Move list** — scrolling, color-coded by move quality (blunder = red, brilliant = teal, etc.).
- **Commentary** — engine annotation or your own hand-written notes loaded from a text file.
- **Metrics strip** — four plots revealed one move at a time, auto-scaled to the game's actual data range, with the range shown in the title.

---

## Requirements

| Dependency | Notes |
|---|---|
| Python 3.10+ | |
| [Manim Community](https://www.manim.community/) v0.20+ | `pip install manim` |
| [manim-chess](https://github.com/Spijkervet/manim-chess) | Provides `Board` and `EvaluationBar` |
| [python-chess](https://python-chess.readthedocs.io/) | `pip install chess` |
| [Stockfish](https://stockfishchess.org/download/) | Binary on your system |

Install Python dependencies:

```bash
pip install manim chess
# install manim-chess per its own instructions
```

---

## File Overview

| File | Purpose |
|---|---|
| `run_animator.py` | **Start here.** CLI entry point — runs analysis and/or Manim. |
| `animator_game.py` | Main Manim scene (`AnimatedGame`). Move loop, panels, metrics. |
| `animator_layout.py` | All geometry constants, colors, and fonts in one place. |
| `animator_initial_frame.py` | Initial frame scene and `GameInfo` dataclass. |
| `animator_metrics.py` | Four-plot metrics strip (Eval, Space, Mobility, King Safety). |
| `chess_game_analyzer.py` | Stockfish wrapper — produces per-move positional metrics. |
| `convert_script_to_comment_dict.py` | Parses a `[KEY]` commentary text file into a dict. |
| `evaluation_bar.py` | `EvaluationBar` Mobject (part of manim-chess). |

---

## Quick Start

The repository includes `sample_game.pgn` — Caruana vs. Nepomniachtchi, Round 5 of the 2024 Candidates Tournament (Toronto), an Italian Game ending in a draw by repetition after 32 moves. All examples below use this file.

### 1. Analyze and animate in one step

```bash
python run_animator.py sample_game --analyze --depth 20
```

This runs Stockfish at depth 20, saves `sample_game_analysis.json`, then renders a low-quality preview video. The `--analyze` flag is only needed the first time; subsequent renders reuse the saved JSON.

### 2. Re-render without re-analyzing

```bash
python run_animator.py sample_game
```

### 3. Quality and resolution

Manim's quality flag controls both resolution and frame rate together:

| Flag | Resolution | FPS | Use case |
|---|---|---|---|
| `--quality low` | 854 × 480 | 15 | Fast preview during development |
| `--quality medium` | 1280 × 720 | 30 | Draft review |
| `--quality high` | 1920 × 1080 | 60 | Final YouTube/Vimeo upload |
| `--quality ultra` | 3840 × 2160 | 60 | 4K archival render |

```bash
# Fast preview (default)
python run_animator.py sample_game --quality low

# 1080p final render, no auto-open
python run_animator.py sample_game --quality high --no-preview

# 4K archival render (slow — allow 30–60 min on a typical machine)
python run_animator.py sample_game --quality ultra --no-preview
```

### 4. Custom Stockfish path

```bash
python run_animator.py sample_game --analyze --depth 22 \
    --stockfish /opt/homebrew/bin/stockfish
```

Default path is `/usr/local/bin/stockfish`. On Apple Silicon Macs installed via Homebrew the binary is typically at `/opt/homebrew/bin/stockfish`.

### 5. Your own game

Replace `sample_game` with any base filename. The script looks for `{name}.pgn`, `{name}_analysis.json`, and optionally `{name}_notes.txt` in the current directory:

```bash
python run_animator.py my_game --analyze --depth 20
```

---

## Adding Your Own Commentary

Create a plain text file named `sample_game_notes.txt` (or `{game_id}_notes.txt` for your own game) in the same directory. Each entry is a ply number in square brackets — where ply 1 = White's first move, ply 2 = Black's first move, and so on — followed by your comment:

```
[1] Caruana opens with the King's Pawn, staking a central claim immediately.

[10] The Italian Game — one of the oldest and most deeply analyzed openings in chess.

[23] A key moment: after the exchange on e3, White's rook structure becomes more active.

[47] Repetition begins. White has a slight edge but Black holds the balance.
```

Comments are word-wrapped automatically to fit the commentary panel. If no notes file is present, the panel falls back to engine annotations (move classification, centipawn loss, current evaluation).

---

## How the Analysis Works

`chess_game_analyzer.py` runs Stockfish on each position and also computes four positional metrics directly from the board using `python-chess`:

| Metric | What it measures |
|---|---|
| **Eval** | Stockfish centipawn evaluation, converted to pawns |
| **Space** | Control of central territory, weighted by piece count behind the pawn chain |
| **Mobility** | Legal moves per piece type, weighted and penalized for unsafe squares |
| **King Safety** | Pawn shield strength, king tropism, and attack units near the king |

The metrics strip auto-scales each plot to the actual range of values in the game (with 15% padding) so variation is always visible regardless of the absolute values. The range is shown next to each plot title, e.g. `King Safety [-0.22, 1.7]`.

---

## Testing Without a Game File

A `QuickDemo` scene is included that runs without any PGN or analysis file:

```bash
python run_animator.py --scene QuickDemo
# or directly:
manim -pql animator_game.py QuickDemo
```

The metrics strip can also be tested independently with synthetic sine-wave data:

```bash
manim -pql animator_metrics.py MetricsDebug
```

---

## Customization

### Changing the eval bar scale

The eval bar is currently scaled to ±4 pawns. To change it, edit `ScaledEvaluationBar._SLOPE` in `animator_game.py`:

```python
# slope = 0.737063 × (10 / your_range)
_SLOPE = 1.8427   # ±4 pawns
_SLOPE = 0.9238   # ±8 pawns
_SLOPE = 0.737063 # ±10 pawns (original)
```

Also update the clamp in the animation loop:

```python
eval_pawns = max(-4.0, min(4.0, move.eval_after / 100.0))
```

### Swapping the fourth metrics plot

The fourth plot is King Safety. To swap it for Threats (or FTI), edit the `_ks_w` / `_ks_b` extraction in `MetricPlotPanel.__init__` in `animator_metrics.py` and the corresponding `advance_to_move()` call. The fields available on each `MoveData` are: `space_white/black`, `mobility_white/black`, `king_safety_white/black`, `threats_white/black`, `fti1`, `fti2`, `fti3`.

### Colors and fonts

All colors and font sizes are in `animator_layout.py` — `ColorScheme` and `Typography` dataclasses at the top of the file.

---

## Example Output

> **Screenshot:** To add an illustrative image to this README, render the sample game at low quality, then extract a still from the output video using ffmpeg:
> ```bash
> python run_animator.py sample_game --analyze --depth 20
> # Video is saved to media/videos/animator_game/480p15/AnimatedGame.mp4
> ffmpeg -i media/videos/animator_game/480p15/AnimatedGame.mp4 \
>        -ss 00:00:10 -vframes 1 docs/screenshot.png
> ```
> Then add to this README:
> ```markdown
> ![Sample frame](docs/screenshot.png)
> ```

---

## Data Flow

```
sample_game.pgn                          (included in repository)
    │
    ▼
chess_game_analyzer.py                   (--analyze flag, run once)
    │
    ├──► sample_game_analysis.json       (reused on subsequent renders)
    └──► sample_game_notes.txt           (optional, hand-written)

run_animator.py
    ├── writes sample_game_animator_config.json
    ├── sets CHESS_ANIMATOR_CONFIG environment variable
    └── calls: manim -pql animator_game.py AnimatedGame

AnimatedGame.construct()
    ├── loads analysis JSON  →  List[MoveData]
    ├── builds board, eval bar, header, move list, commentary, metrics strip
    └── for each move:
            animate board position
            update eval bar
            update move list (scrolling)
            update commentary
            reveal next metrics segment
```

---

## License

MIT License. See `LICENSE` for details.

---

## Author

David Joyner
