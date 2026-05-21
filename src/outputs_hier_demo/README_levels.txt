Hierarchy used in this demo:
- Global: whole image ↔ whole caption.
- Entities: detected objects (image) ↔ noun chunks (text).
- Relations: spatial relations between object pairs (image) ↔ subject–relation–object phrases (text).

Alignment: initial entity correspondences via Hungarian on cosine similarity (text↔image),
plus a global anchor; then Orthogonal Procrustes maps text embeddings into the image space.
BEFORE/AFTER plots visualize gap reduction per level.
