"""Check whether color 9 (our inferred avatar color) forms a single
connected blob, or multiple disconnected regions -- which would corrupt
centroid-based position tracking. Pure numpy/BFS, no scipy needed."""
import json
import sys
from collections import deque
import numpy as np

AVATAR_COLOR = 9

def load_frames(path):
    frames = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)["data"]
            if "frame" not in d or not d["frame"]:
                continue
            frames.append(np.array(d["frame"][0]))
    return frames

def count_blobs(mask):
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    n_blobs = 0
    sizes = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not visited[y, x]:
                n_blobs += 1
                size = 0
                q = deque([(y, x)])
                visited[y, x] = True
                while q:
                    cy, cx = q.popleft()
                    size += 1
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny,nx] and not visited[ny,nx]:
                            visited[ny,nx] = True
                            q.append((ny,nx))
                sizes.append(size)
    return n_blobs, sizes

def analyze(path):
    frames = load_frames(path)
    multi_blob_count = 0
    blob_counts = []
    example_sizes = None
    for f in frames:
        mask = (f == AVATAR_COLOR)
        n, sizes = count_blobs(mask)
        blob_counts.append(n)
        if n > 1 and example_sizes is None:
            example_sizes = sizes
        if n > 1:
            multi_blob_count += 1
    print(f"{path.split('/')[-1]}: {len(frames)} frames, "
          f"blob-count distribution={sorted(set(blob_counts))}, "
          f"frames with >1 blob: {multi_blob_count}/{len(frames)}, "
          f"example multi-blob sizes: {example_sizes}")

for p in sys.argv[1:]:
    analyze(p)
