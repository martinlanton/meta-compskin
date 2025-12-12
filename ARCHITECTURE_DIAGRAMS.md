# Architecture Diagrams - metacompskin

This document contains Mermaid diagrams visualizing the architecture, data flow, and key algorithms of the `metacompskin` package.

---

## 1. Package Architecture Overview

```mermaid
graph TB
    subgraph "Input/Output"
        INPUT[📁 Input NPZ<br/>Blendshapes]
        OUTPUT[📁 Output NPZ<br/>Compressed Model]
        FRAMES[📁 OBJ Files<br/>Animation Frames]
    end

    subgraph "Main Components"
        DATA[📊 BlendshapeModelData<br/>Loads & validates data]
        COMPRESS[🔧 SkinCompressor<br/>Learns compact representation]
        ANIMATE[🎬 AnimationFrameGenerator<br/>Generates animation]
    end

    subgraph "Helpers"
        UTILS[⚙️ Utilities<br/>Math helpers]
    end

    INPUT --> DATA
    DATA --> COMPRESS
    COMPRESS --> OUTPUT
    OUTPUT --> ANIMATE
    ANIMATE --> FRAMES

    COMPRESS -.-> UTILS
    ANIMATE -.-> UTILS

    classDef io fill:#FFD93D,stroke:#CCB030,color:#000
    classDef main fill:#4A90E2,stroke:#2E5C8A,color:#fff
    classDef helper fill:#90EE90,stroke:#70CE70,color:#000

    class INPUT,OUTPUT,FRAMES io
    class DATA,COMPRESS,ANIMATE main
    class UTILS helper
```

---

## 2. Class Diagram - Core Classes

```mermaid
classDiagram
    class BlendshapeModelData {
        <<frozen dataclass>>
        +ndarray deltas
        +ndarray rest_verts
        +ndarray rest_faces
        +dict inbetween_info
        +dict combination_info
        +str model_name
        +float alpha
        +int n_blendshapes
        +int n_vertices
        +int n_faces
        +from_npz(path) BlendshapeModelData$
        +print_details() void
        -_validate_arrays(...) void$
    }
    
    class SkinCompressor {
        +BlendshapeModelData model_data
        +int iterations
        +int number_of_bones
        +int max_influences
        +int total_nnz_B_rt
        +float init_weight
        +int power
        +int seed
        +float alpha
        +str device
        +list loss_list
        +list abserr_list
        +run(output_location) void
        +train(B_rt, TR, A, W, normalizeW) void
        +compBX(Wn, B_rt, TR, n_bs, P) tuple
        +buildTR() Tensor
        +get_matrix_for_optimization() Tensor
        +get_laplacian_regularization(rest_faces) Tensor
    }
    
    class AnimationFrameGenerator {
        +Path compressed_data_path
        +BlendshapeModelData model_data
        +ndarray rest_verts
        +ndarray quads
        +ndarray weights
        +ndarray rest_xform
        +ndarray shape_xform
        +int num_bones
        +ndarray rest_pose_homog
        +generate_frames(anim_path, output_dir, max_controls) void
        -_generate_skinning_transforms(weights) ndarray
        -_add_homogeneous_coordinate(M, dim) ndarray$
    }
    
    class Utils {
        <<module>>
        +add_homogeneous_coordinate(M, dim) ndarray$
        +npf(T) ndarray$
    }
    
    class Constants {
        <<module>>
        +dict MODEL_ALPHA_VALUES$
        +float DEFAULT_ALPHA_VALUE$
        +get_alpha_for_model(name) float$
    }
    
    SkinCompressor --> BlendshapeModelData : uses
    SkinCompressor --> Utils : uses
    SkinCompressor --> Constants : uses
    AnimationFrameGenerator --> BlendshapeModelData : uses
    AnimationFrameGenerator --> Utils : uses
    BlendshapeModelData --> Constants : uses
```

---

## 3. Data Flow - Compression Pipeline

```mermaid
flowchart TD
    START([Start]) --> LOAD[Load NPZ File]
    LOAD --> VALIDATE[Validate Arrays<br/>BlendshapeModelData]
    
    VALIDATE --> INIT[Initialize SkinCompressor<br/>Set parameters:<br/>P=40 bones<br/>K=8 influences<br/>L=6000 non-zeros]
    
    INIT --> GET_A[Get Matrix A<br/>Reshape deltas<br/>3S × N]
    GET_A --> GET_L[Compute Laplacian<br/>Regularization Matrix]
    GET_L --> BUILD_TR[Build TR Basis<br/>6-DOF transformation]
    
    BUILD_TR --> INIT_PARAMS[Initialize Parameters<br/>B_rt: random 6×S×P<br/>W: random P×N]
    
    INIT_PARAMS --> PHASE1{Phase 1<br/>normalizeW=False}
    PHASE1 --> TRAIN1[Train 10k iterations<br/>Adam + Projection]
    
    TRAIN1 --> PHASE2{Phase 2<br/>normalizeW=True}
    PHASE2 --> TRAIN2[Train 10k iterations<br/>Adam + Projection]
    
    TRAIN2 --> NORMALIZE[Normalize Weights<br/>W_n = W / Σ W]
    
    NORMALIZE --> EVAL[Evaluate Errors<br/>MAE, MXE]
    
    EVAL --> SAVE[Save NPZ<br/>rest, quads, weights<br/>restXform, shapeXform]
    
    SAVE --> END([End])
    
    style START fill:#90EE90
    style END fill:#FFB6C1
    style PHASE1 fill:#FFD93D
    style PHASE2 fill:#FFD93D
    style TRAIN1 fill:#4A90E2,color:#fff
    style TRAIN2 fill:#4A90E2,color:#fff
```

---

## 4. Data Flow - Animation Generation

```mermaid
flowchart TD
    START([Start]) --> LOAD_COMP[Load Compressed NPZ<br/>weights, shapeXform]
    LOAD_COMP --> LOAD_MODEL[Load Model Data<br/>inbetween, corrective info]
    
    LOAD_MODEL --> LOAD_ANIM[Load Animation NPZ<br/>control weights]
    
    LOAD_ANIM --> LOOP_START{For each frame}
    
    LOOP_START --> RIG[Apply Rig Logic<br/>72 controls → S blendshapes]
    
    RIG --> GEN_XFORM[Generate Skinning Transforms<br/>Equation 7: M_j = I + Σ c_k N_k,j]
    
    GEN_XFORM --> APPLY_SKIN[Apply Skinning<br/>X = weights ⊗ rest_pose<br/>result = M @ X]
    
    APPLY_SKIN --> SAVE_OBJ[Save OBJ File<br/>frame_XXXXX.obj]
    
    SAVE_OBJ --> LOOP_END{More frames?}
    LOOP_END -->|Yes| LOOP_START
    LOOP_END -->|No| END([End])
    
    style START fill:#90EE90
    style END fill:#FFB6C1
    style RIG fill:#7B68EE,color:#fff
    style GEN_XFORM fill:#4A90E2,color:#fff
    style APPLY_SKIN fill:#4A90E2,color:#fff
```

---

## 5. Training Loop - Proximal Adam Algorithm

```mermaid
flowchart TD
    START([Start Training]) --> INIT_OPT[Initialize Adam Optimizer<br/>lr=1e-3, β=(0.9, 0.9)]
    
    INIT_OPT --> LOOP{For i in<br/>iterations}
    
    LOOP --> NORM_W{normalizeW?}
    NORM_W -->|True| W_NORM[W_n = W / Σ W]
    NORM_W -->|False| W_RAW[W_n = W]
    
    W_NORM --> FORWARD
    W_RAW --> FORWARD[Forward Pass<br/>B·C = compBX]
    
    FORWARD --> LOSS[Compute Loss<br/>L_p + α·Laplacian]
    
    LOSS --> BACKWARD[Backward Pass<br/>Compute Gradients]
    
    BACKWARD --> ADAM[Adam Step<br/>Update B_rt, W]
    
    ADAM --> PROJ_START[Proximal Projection]
    
    PROJ_START --> PROJ_W[Project W:<br/>1. Keep K largest per vertex<br/>2. Clamp to non-negative]
    
    PROJ_W --> PROJ_B[Project B_rt:<br/>Keep L largest globally]
    
    PROJ_B --> LOG{i % 200 == 0?}
    LOG -->|Yes| PRINT[Print Progress<br/>Loss, Error, Sparsity]
    LOG -->|No| CHECK
    PRINT --> CHECK
    
    CHECK{i < iterations?}
    CHECK -->|Yes| LOOP
    CHECK -->|No| END([End Training])
    
    style START fill:#90EE90
    style END fill:#FFB6C1
    style PROJ_W fill:#FF6B6B,color:#fff
    style PROJ_B fill:#FF6B6B,color:#fff
    style ADAM fill:#4A90E2,color:#fff
```

---

## 6. Matrix Decomposition - compBX

```mermaid
flowchart LR
    subgraph "Input"
        W_N[W_n<br/>P × N<br/>Weights]
        B_RT[B_rt<br/>6 × S × P<br/>Parameters]
        TR[TR<br/>6 × 1 × 1 × 3 × 4<br/>Basis]
        REST[rest_pose<br/>N × 4<br/>Vertices]
    end
    
    subgraph "Compute C"
        W_N --> UNSQUEEZE[unsqueeze<br/>P × 1 × N]
        REST --> BROADCAST
        UNSQUEEZE --> MUL[Element-wise<br/>Multiply]
        BROADCAST[broadcast] --> MUL
        MUL --> PERMUTE[permute<br/>P × 4 × N]
        PERMUTE --> RESHAPE_C[reshape<br/>4P × N]
        RESHAPE_C --> C[C Matrix<br/>4P × N]
    end
    
    subgraph "Compute B"
        B_RT --> LOOP[Loop i=0..5:<br/>Sum B_rt[i] * TR[i]]
        TR --> LOOP
        LOOP --> PERMUTE_B[permute<br/>S × 3 × P × 4]
        PERMUTE_B --> RESHAPE_B[reshape<br/>3S × 4P]
        RESHAPE_B --> B[B Matrix<br/>3S × 4P]
    end
    
    subgraph "Output"
        B --> MATMUL[Matrix<br/>Multiply]
        C --> MATMUL
        MATMUL --> BX[BX = B @ C<br/>3S × N]
        B --> OUT_B[Return B]
        C --> OUT_C[Return X]
        BX --> OUT_BX[Return BX]
    end
    
    style C fill:#4A90E2,color:#fff
    style B fill:#7B68EE,color:#fff
    style BX fill:#50C878,color:#fff
```

---

## 7. Transformation Basis - buildTR

```mermaid
graph TD
    subgraph "6-DOF Transformation Basis"
        TR1["TR[0]<br/>Rotation around Z<br/>[ 0  0  0  0]<br/>[ 0  0 -1  0]<br/>[ 0  1  0  0]"]
        TR2["TR[1]<br/>Rotation around Y<br/>[ 0  0  1  0]<br/>[ 0  0  0  0]<br/>[-1  0  0  0]"]
        TR3["TR[2]<br/>Rotation around X<br/>[ 0 -1  0  0]<br/>[ 1  0  0  0]<br/>[ 0  0  0  0]"]
        TR4["TR[3]<br/>Translation X<br/>[ 0  0  0  1]<br/>[ 0  0  0  0]<br/>[ 0  0  0  0]"]
        TR5["TR[4]<br/>Translation Y<br/>[ 0  0  0  0]<br/>[ 0  0  0  1]<br/>[ 0  0  0  0]"]
        TR6["TR[5]<br/>Translation Z<br/>[ 0  0  0  0]<br/>[ 0  0  0  0]<br/>[ 0  0  0  1]"]
    end
    
    subgraph "Combination"
        COMBINE["N_k,j = Σ B_rt[i,k,j] * TR[i]<br/><br/>Result: 3×4 matrix<br/>[ 0   -r3   r2   t1]<br/>[ r3   0   -r1   t2]<br/>[-r2   r1   0    t3]"]
    end
    
    subgraph "Properties"
        PROP1[Linearized Rotation<br/>Skew-symmetric 3×3]
        PROP2[Linear Blending<br/>Closed under addition]
        PROP3[Fast Evaluation<br/>No quaternion slerp]
    end
    
    TR1 --> COMBINE
    TR2 --> COMBINE
    TR3 --> COMBINE
    TR4 --> COMBINE
    TR5 --> COMBINE
    TR6 --> COMBINE
    
    COMBINE --> PROP1
    COMBINE --> PROP2
    COMBINE --> PROP3
    
    style TR1 fill:#FFB6C1
    style TR2 fill:#FFB6C1
    style TR3 fill:#FFB6C1
    style TR4 fill:#B0E0E6
    style TR5 fill:#B0E0E6
    style TR6 fill:#B0E0E6
    style COMBINE fill:#50C878,color:#fff
```

---

## 8. Sequence Diagram - Full Workflow

```mermaid
sequenceDiagram
    participant User
    participant BlendshapeModelData
    participant SkinCompressor
    participant Adam as Adam Optimizer
    participant AnimationFrameGenerator
    participant Output
    
    User->>BlendshapeModelData: from_npz("aura.npz")
    BlendshapeModelData->>BlendshapeModelData: validate arrays
    BlendshapeModelData->>BlendshapeModelData: extract model_name, alpha
    BlendshapeModelData-->>User: model_data
    
    User->>SkinCompressor: __init__(model_data, iterations=10k)
    SkinCompressor->>SkinCompressor: initialize parameters
    SkinCompressor->>SkinCompressor: set device (cuda/cpu)
    SkinCompressor-->>User: compressor
    
    User->>SkinCompressor: run("output.npz")
    
    SkinCompressor->>SkinCompressor: get_matrix_for_optimization()
    SkinCompressor->>SkinCompressor: get_laplacian_regularization()
    SkinCompressor->>SkinCompressor: buildTR()
    SkinCompressor->>SkinCompressor: initialize B_rt, W
    
    loop Phase 1: normalizeW=False (10k iterations)
        SkinCompressor->>SkinCompressor: compBX()
        SkinCompressor->>SkinCompressor: compute loss
        SkinCompressor->>Adam: backward() + step()
        Adam-->>SkinCompressor: updated params
        SkinCompressor->>SkinCompressor: proximal projection
    end
    
    loop Phase 2: normalizeW=True (10k iterations)
        SkinCompressor->>SkinCompressor: compBX()
        SkinCompressor->>SkinCompressor: compute loss
        SkinCompressor->>Adam: backward() + step()
        Adam-->>SkinCompressor: updated params
        SkinCompressor->>SkinCompressor: proximal projection
    end
    
    SkinCompressor->>SkinCompressor: normalize weights
    SkinCompressor->>SkinCompressor: evaluate errors
    SkinCompressor->>Output: save NPZ (weights, shapeXform)
    SkinCompressor-->>User: done
    
    User->>AnimationFrameGenerator: __init__(compressed.npz, model_data)
    AnimationFrameGenerator->>AnimationFrameGenerator: load compressed data
    AnimationFrameGenerator-->>User: generator
    
    User->>AnimationFrameGenerator: generate_frames(anim.npz, "frames/")
    
    loop For each frame
        AnimationFrameGenerator->>AnimationFrameGenerator: apply rig logic
        AnimationFrameGenerator->>AnimationFrameGenerator: generate transforms (Eq 7)
        AnimationFrameGenerator->>AnimationFrameGenerator: apply skinning
        AnimationFrameGenerator->>Output: save OBJ file
    end
    
    AnimationFrameGenerator-->>User: done
```

---

## 9. State Machine - Training States

```mermaid
stateDiagram-v2
    [*] --> Initialized: SkinCompressor.__init__()
    
    Initialized --> Phase1: run() called
    
    state Phase1 {
        [*] --> ComputeForward
        ComputeForward --> ComputeLoss
        ComputeLoss --> Backward
        Backward --> AdamUpdate
        AdamUpdate --> ProximalProject
        ProximalProject --> LogProgress: i % 200 == 0
        ProximalProject --> CheckIterations: i % 200 != 0
        LogProgress --> CheckIterations
        CheckIterations --> ComputeForward: i < 10000
        CheckIterations --> [*]: i == 10000
    }
    
    Phase1 --> Phase2: Phase 1 complete
    
    state Phase2 {
        [*] --> ComputeForward2: normalizeW=True
        ComputeForward2 --> ComputeLoss2
        ComputeLoss2 --> Backward2
        Backward2 --> AdamUpdate2
        AdamUpdate2 --> ProximalProject2
        ProximalProject2 --> LogProgress2: i % 200 == 0
        ProximalProject2 --> CheckIterations2: i % 200 != 0
        LogProgress2 --> CheckIterations2
        CheckIterations2 --> ComputeForward2: i < 10000
        CheckIterations2 --> [*]: i == 10000
    }
    
    Phase2 --> Finalize: Phase 2 complete
    
    state Finalize {
        [*] --> NormalizeWeights
        NormalizeWeights --> EvaluateErrors
        EvaluateErrors --> SaveResults
        SaveResults --> [*]
    }
    
    Finalize --> [*]: NPZ saved
```

---

## 10. Component Interaction - Runtime

```mermaid
graph TB
    subgraph "Offline Pre-processing (Hours)"
        INPUT_NPZ[Input NPZ<br/>Blendshapes<br/>S shapes, N vertices]
        COMPRESS[SkinCompressor<br/>Optimization<br/>20k iterations]
        OUTPUT_NPZ[Output NPZ<br/>Compressed<br/>~90% sparse]
        
        INPUT_NPZ --> COMPRESS
        COMPRESS --> OUTPUT_NPZ
    end
    
    subgraph "Runtime Per-Frame (Microseconds)"
        RIG_INPUT[Rig Controls<br/>~72 parameters]
        RIG_LOGIC[Rig Logic<br/>Inbetween +<br/>Correctives]
        BLEND_WEIGHTS[Blendshape Weights<br/>S values]
        CPU_XFORM[CPU: Compute M_j<br/>Sparse MatVec<br/>Equation 7]
        GPU_SKIN[GPU: Skinning<br/>Σ w_i,j M_j v_0,i<br/>Equation 2]
        OUTPUT_MESH[Output Mesh<br/>N vertices]
        
        RIG_INPUT --> RIG_LOGIC
        RIG_LOGIC --> BLEND_WEIGHTS
        BLEND_WEIGHTS --> CPU_XFORM
        OUTPUT_NPZ -.->|shapeXform| CPU_XFORM
        CPU_XFORM --> GPU_SKIN
        OUTPUT_NPZ -.->|weights, rest| GPU_SKIN
        GPU_SKIN --> OUTPUT_MESH
    end
    
    style INPUT_NPZ fill:#FFD93D
    style OUTPUT_NPZ fill:#50C878,color:#fff
    style COMPRESS fill:#4A90E2,color:#fff
    style CPU_XFORM fill:#FF6B6B,color:#fff
    style GPU_SKIN fill:#7B68EE,color:#fff
    style OUTPUT_MESH fill:#90EE90
```

---

## 11. Memory Layout - Tensor Shapes

```mermaid
graph TD
    subgraph "Input Data"
        DELTAS["deltas<br/>(S, N, 3)<br/>Blendshape deltas"]
        REST_V["rest_verts<br/>(N, 3)<br/>Rest pose"]
        REST_F["rest_faces<br/>(F, 4)<br/>Quad faces"]
    end
    
    subgraph "Optimization Variables"
        B_RT["B_rt<br/>(6, S, P, 1, 1)<br/>6-DOF parameters<br/>LEARNABLE"]
        W["W<br/>(P, N)<br/>Skinning weights<br/>LEARNABLE"]
    end
    
    subgraph "Fixed Basis"
        TR["TR<br/>(6, 1, 1, 3, 4)<br/>Transform basis<br/>FIXED"]
    end
    
    subgraph "Intermediate"
        A["A<br/>(3S, N)<br/>Target matrix"]
        B["B<br/>(3S, 4P)<br/>Transform matrix"]
        C["C<br/>(4P, N)<br/>Weighted rest"]
        BX["BX<br/>(3S, N)<br/>Approximation"]
    end
    
    subgraph "Output"
        W_FINAL["weights<br/>(N, P)<br/>Normalized"]
        SHAPE_X["shapeXform<br/>(3S, 4P)<br/>Sparse B"]
    end
    
    DELTAS --> A
    REST_V --> C
    
    B_RT --> B
    TR --> B
    W --> C
    
    B --> BX
    C --> BX
    
    BX --> LOSS[Loss Function]
    A --> LOSS
    
    W --> W_FINAL
    B --> SHAPE_X
    
    style B_RT fill:#FF6B6B,color:#fff
    style W fill:#FF6B6B,color:#fff
    style TR fill:#B0E0E6
    style BX fill:#50C878,color:#fff
```

---

## Notes

- **S**: Number of blendshapes
- **N**: Number of vertices
- **P**: Number of proxy bones (default 40)
- **K**: Maximum influences per vertex (default 8)
- **F**: Number of faces

All diagrams follow the paper notation and reference equations where applicable.

