# AI Agent Guidelines for Multi-UAV-TA

This repository has recently transitioned to Version 02.0. Follow these project-scoped rules for all modifications:

1. **Strict Architecture Boundaries**: 
   - `mUAV_TA/`: Core environment logic only (PettingZoo/Gymnasium). **Never add UI, PyGame, or rendering code here.**
   - `core_sim/`: Rust physics backend. Use this for O(N^2) math, distance matrices, and collision detection. Compile with `maturin develop`.
   - `server/api.py`: FastAPI backend that broadcasts state.
   - `frontend/`: React Three Fiber frontend. **All visualization must happen here.**

2. **Performance First**:
   When implementing new simulation features, always evaluate if the code will block RL training. If it's a slow Python loop, move it to the Rust module (`core_sim`).

3. **Dependencies**:
   - Do not use `gym` (legacy). Use `gymnasium`.
   - Ensure the `.venv` is active when running Python commands.

For more detailed architectural notes, read `AI_DEVELOPMENT_GUIDE.md` in the root of the project.
