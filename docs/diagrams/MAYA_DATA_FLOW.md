# Maya Data Flow - OBJ to Compressed Model

This document shows the complete data flow from OBJ files through Maya loading, compression, and final output.

---

## Complete Pipeline Sequence

```mermaid
sequenceDiagram
    participant User
    participant API as MayaBlendshapeModelData
    participant Loader as maya_loader
    participant Maya as mayapy subprocess
    participant Compressor as SkinCompressor
    participant Generator as AnimationFrameGenerator
    
    User->>API: from_obj_files(rest, shapes, mayapy)
    
    Note over API,Loader: Load Rest Mesh
    API->>Loader: load_obj_with_maya(HEAD.obj, mayapy)
    Loader->>Maya: Execute loading script
    Maya->>Maya: Import OBJ<br/>Extract vertices & faces
    Maya-->>Loader: JSON: {vertices, faces}
    Loader-->>API: (rest_verts, rest_faces)
    
    Note over API,Loader: Load Blendshapes
    loop For each blendshape OBJ
        API->>Loader: load_obj_with_maya(shape.obj, mayapy)
        Loader->>Maya: Execute loading script
        Maya->>Maya: Import OBJ<br/>Extract geometry
        Maya-->>Loader: JSON: {vertices, faces}
        Loader-->>API: (shape_verts, shape_faces)
        API->>API: Validate topology<br/>(same vertex/face count)
    end
    
    Note over API: Compute Deltas
    API->>API: deltas = shape_verts - rest_verts
    API->>API: Validate arrays
    API-->>User: BlendshapeModelData(deltas, rest, faces, ...)
    
    Note over User,Compressor: Compression Phase
    User->>Compressor: SkinCompressor(model_data, iterations=600)
    User->>Compressor: run(output_location)
    Compressor->>Compressor: Optimize 6-DOF transforms<br/>(Equation 7)
    Compressor->>Compressor: Compute sparse weights<br/>(max K per vertex)
    Compressor->>Compressor: Save: rest, weights, restXform, shapeXform
    Compressor-->>User: Compressed NPZ file
    
    Note over User,Generator: Animation Generation (Optional)
    User->>Generator: AnimationFrameGenerator(compressed_npz, model_data)
    User->>Generator: generate_frames(animation_weights, output_dir)
    Generator->>Generator: For each frame:<br/>Apply rig logic<br/>Compute skinning transforms<br/>Apply LBS
    Generator-->>User: OBJ frame files
```

---

## Data Structures at Each Stage

### 1. Input - OBJ Files
```
HEAD.obj          → rest mesh (neutral pose)
AU1.obj           → blendshape target 1
AU2.obj           → blendshape target 2
ShapeJawOpen.obj  → blendshape target 3
```

### 2. Loaded Geometry
```python
rest_verts: NDArray[float64]    # Shape: (N, 3)
rest_faces: NDArray[int32]      # Shape: (F, 4) - quads
shape_verts: List[NDArray]      # List of (N, 3) arrays
```

### 3. Computed Deltas
```python
deltas: NDArray[float64]        # Shape: (S, N, 3)
# Where S = number of blendshapes
#       N = number of vertices
```

### 4. BlendshapeModelData
```python
BlendshapeModelData(
    deltas: (S, N, 3),
    rest_verts: (N, 3),
    rest_faces: (F, 4),
    inbetween_info: dict,
    combination_info: dict,
    model_name: str,
    alpha: float
)
```

### 5. Compressed Output (NPZ)
```python
{
    'rest': (N, 3),              # Rest vertices
    'quads': (F, 4),             # Face connectivity
    'weights': (N, P),           # Sparse skinning weights
    'restXform': (3, 4*P),       # Rest transformations
    'shapeXform': (3*S, 4*P)     # Shape transformations (sparse)
}
# Where P = number of bones (typically 40)
```

---

## Timeline and Performance

| Stage | Duration | Notes |
|-------|----------|-------|
| **Load REST.obj** | ~5s | Subprocess startup + Maya init |
| **Load 3 shapes** | ~15s | 3 × ~5s per shape |
| **Compute deltas** | <0.1s | Vectorized numpy operation |
| **Validate topology** | <0.1s | Simple array shape checks |
| **Create model_data** | <0.1s | Dataclass instantiation |
| **Compression (600 iter)** | ~20s | GPU-accelerated optimization |
| **Total** | **~40s** | For 3 shapes, 5761 vertices |

### Scaling
- Loading time: ~5s per OBJ (linear)
- Compression time: ~10-30s (depends on iterations, not shape count)
- For 70 shapes: ~350s loading + ~30s compression = **~6.5 minutes**

---

## Error Handling Points

The pipeline includes validation at multiple stages:

```mermaid
flowchart LR
    INPUT[OBJ Files] -->|File exists?| CHECK1{✓}
    CHECK1 -->|No| ERR1[FileNotFoundError]
    CHECK1 -->|Yes| LOAD[Maya Loading]
    
    LOAD -->|Import success?| CHECK2{✓}
    CHECK2 -->|No| ERR2[RuntimeError]
    CHECK2 -->|Yes| TOPO[Topology Check]
    
    TOPO -->|Same verts/faces?| CHECK3{✓}
    CHECK3 -->|No| ERR3[ValueError:<br/>Topology mismatch]
    CHECK3 -->|Yes| DELTA[Compute Deltas]
    
    DELTA --> VALIDATE[Array Validation]
    VALIDATE -->|Valid?| CHECK4{✓}
    CHECK4 -->|No| ERR4[ValueError:<br/>Invalid arrays]
    CHECK4 -->|Yes| OUTPUT[BlendshapeModelData]
    
    classDef errorStyle fill:#FFE5E5,stroke:#FF0000
    classDef successStyle fill:#E5FFE5,stroke:#00FF00
    
    class ERR1,ERR2,ERR3,ERR4 errorStyle
    class OUTPUT successStyle
```

---

## Integration with Existing Pipeline

The Maya loading system integrates seamlessly with the existing compression pipeline:

```
[Maya OBJ Files]
       ↓
[MayaBlendshapeModelData.from_obj_files()]  ← NEW
       ↓
[BlendshapeModelData]                       ← Existing interface
       ↓
[SkinCompressor]                            ← Existing
       ↓
[Compressed NPZ]                            ← Existing
       ↓
[AnimationFrameGenerator]                   ← Existing
       ↓
[OBJ Frame Files]                           ← Existing
```

**Key Design Decision**: MayaBlendshapeModelData returns standard BlendshapeModelData, so all downstream code works without modification.
