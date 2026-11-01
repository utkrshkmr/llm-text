# Lab B: Accelerator models (Appendix B)

Cycle-level systolic-array simulators (output- and weight-stationary), the tiling/padding
utilisation model, the 32x32-tile / 16x16-face layout used by Tenstorrent Tensix cores,
Cannon/SUMMA matmul on a mesh of cores with link-traffic accounting, and roofline/decode bounds.
Pure NumPy.

```bash
cd code
LAB_IMPL=starter  python -m pytest -q labs/labB_accel
LAB_IMPL=solution python -m pytest -q labs/labB_accel
python labs/labB_accel/report.py
```
Reference implementation: `llmsys/systolic.py`.
