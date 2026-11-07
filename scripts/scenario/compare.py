"""Use golden_frame_test's existing comparison implementation without launching a game."""
import importlib.util
from pathlib import Path


def compare_frame(actual, reference, tolerance):
    path = Path(__file__).resolve().parents[1] / 'golden_frame_test.py'
    spec = importlib.util.spec_from_file_location('scenario_golden_frame', path)
    golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden)
    actual, reference = Path(actual), Path(reference)
    if not actual.is_file() or not reference.is_file():
        raise ValueError(f'missing actual/reference image: {actual}, {reference}')
    pixels, _ = golden.downscale_image(str(actual))
    baseline = golden.load_reference(str(reference))
    if len(pixels) != len(baseline):
        raise ValueError(f'reference must be {golden.TARGET_WIDTH}x{golden.TARGET_HEIGHT} RGB; got {len(baseline)} bytes instead of {len(pixels)}')
    mae = golden.compute_mae(pixels, baseline)
    return dict(passed=mae <= tolerance, mae=mae, tolerance=tolerance,
                actual=str(actual), reference=str(reference))
