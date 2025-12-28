#These constants should be used across all scripts
from src.machine_learn.imports import Path, sp

LEARNING_RATE: float = 0.01
EPOCHS: int = 1_000
PROJECT_DIRECTORY: Path = Path.cwd()
DATA_DIRECTORY: Path = PROJECT_DIRECTORY / 'data' / 'test_data'
X_VARIABLE = sp.symbols('x')