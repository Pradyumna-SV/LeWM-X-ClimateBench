# Pinned upstream LeWorldModel for reproducibility

- **Repository:** https://github.com/lucas-maes/le-wm  
- **Commit:** `bf04d3e8c3752ac24f3692fbc5f4cf50209fa765` (default branch at clone time)

Re-clone or update with:

```bash
cd external/le-wm && git fetch && git checkout bf04d3e8c3752ac24f3692fbc5f4cf50209fa765
```

Training deps (`stable-pretraining`, MuJoCo env extras) target **Python 3.10** per upstream README; Python 3.12 may work for the HDF5 converter alone.
