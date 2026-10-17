"""Optional Streamlit demo (RESEARCH USE ONLY).

    streamlit run inference/app.py -- --checkpoint checkpoints/hybrid/best_model.pth

Streamlit is not part of requirements.txt; install it separately (``pip install streamlit``).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional dependency
    raise SystemExit("Streamlit is not installed. Run `pip install streamlit` to use the demo app.")

from inference.predict import Predictor  # noqa: E402
from utils.common import DISCLAIMER  # noqa: E402


def _checkpoint_arg() -> str:
    args = sys.argv[1:]
    if "--checkpoint" in args and args.index("--checkpoint") + 1 < len(args):
        return args[args.index("--checkpoint") + 1]
    return str(ROOT / "checkpoints" / "hybrid" / "best_model.pth")


@st.cache_resource
def _load(ckpt: str) -> Predictor:
    return Predictor(Path(ckpt), "auto")


def main() -> None:  # pragma: no cover - UI
    st.set_page_config(page_title="DyDA X-ray demo (research only)")
    st.title("DyDA chest X-ray classifier: research demo")
    st.error(DISCLAIMER)
    ckpt = st.text_input("Checkpoint", _checkpoint_arg())
    if not Path(ckpt).is_file():
        st.warning(f"Checkpoint not found: {ckpt}")
        return
    pred = _load(ckpt)
    up = st.file_uploader("Upload a chest X-ray (PNG/JPEG)", type=["png", "jpg", "jpeg"])
    if up is None:
        return
    with tempfile.TemporaryDirectory() as td:
        img = Path(td) / up.name
        img.write_bytes(up.getvalue())
        res = pred.predict(img)
        st.subheader(f"Predicted: {res['predicted_class']} ({res['confidence']:.1%})")
        st.bar_chart(res["probabilities"])
        if st.checkbox("Show Grad-CAM (model sensitivity, not clinical evidence)"):
            out = Path(td) / "cam"
            pred.gradcam(img, out)
            st.image(str(out / "overlay.png"), caption="Grad-CAM overlay")
    st.caption(DISCLAIMER)


main()
