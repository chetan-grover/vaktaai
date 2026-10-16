Record three short webcam clips of yourself speaking (frontal, lit face) and save them here:

- `short.mp4`  ~30 s
- `medium.mp4` ~60 s
- `long.mp4`   ~2 min

They are git-ignored. Tests that need a real face (`real_fixture`) skip until `short.mp4` exists;
everything else runs on a synthetic video generated in `tests/conftest.py`.
