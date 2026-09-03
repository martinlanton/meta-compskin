# Troubleshooting

Symptoms grouped by stage. Each entry says what the message or the behaviour
means and what to do about it.

## Installing

**`ImportError: No module named torch` outside Maya.**
The package was installed with `--no-deps`, or into a different interpreter.
Run `pip install .` in the environment you are using for compression.

**`RuntimeError: Maya modules are not importable` when constructing
`MayaBlendshapeExporter`.**
The exporter only runs inside Maya or `mayapy`. From a plain Python session use
`MayaBlendshapeModelData.from_obj_files` with a `maya_interpreter_path`, or run
the export script through `mayapy`.

**`ModuleNotFoundError: No module named 'numpy'` inside Maya.**
Maya 2023 and older do not bundle numpy: `mayapy -m pip install numpy`.

**The compressor says it is using CPU although there is a GPU.**
The installed torch is the CPU build. `python -c "import torch;
print(torch.__version__)"` should show a `+cu` suffix. Reinstall torch from the
CUDA index before reinstalling the package
([Installation](../getting_started/installation.md#gpu-support)).

## Preparing data

**`KeyError: Missing required keys in NPZ file: [...]`.**
The file lacks one of `deltas`, `rest_verts`, `rest_faces`, `inbetween_info`,
`combination_info`. Write the last two as empty dicts if you have no rig-logic
metadata ([Data formats](../concepts/data_formats.md#input-blendshape-model)).

**`ValueError: deltas must be 3D array (n_blendshapes, n_vertices, 3)` or a
vertex-count mismatch.**
A target does not share the neutral mesh's topology, or the arrays were
stacked along the wrong axis. Check `deltas.shape == (S, N, 3)` and
`rest_verts.shape == (N, 3)`.

**`ValueError: Inconsistent face topology detected` or `Unsupported face
topology`.**
The mesh mixes triangles and quads, or has n-gons. Triangulate or quadrangulate
the whole mesh before exporting. (Note that the rest of your pipeline sees the
same mesh, so do this on the asset, not on a copy.)

**`RuntimeError: ... locked or incoming connections` from the exporter.**
A blendShape target weight is driven by the rig, so the exporter cannot set it.
Export from a scene copy with the driving connections broken, or lay the
targets out as meshes and use `target_meshes=`.

**`ValueError: Topology mismatch detected!` from `from_obj_files`.**
One OBJ has a different vertex or face count. The message names the file.
Re-export all targets from the same scene with the same OBJ options.

**A single shape has an enormous displacement.**
That target was captured with an offset, a transform, or another deformer
active. Fix it in the scene and export again. The check is in
[Preparing data](preparing_data.md#checking-the-input-before-compressing).

**Loading OBJs is very slow.**
Each file spawns `mayapy`, about five seconds of start-up each. Run the loader
inside Maya (no `maya_interpreter_path`) to avoid the spawns, or export with
`MayaBlendshapeExporter` instead.

## Compressing

**`RuntimeError: selected index k out of range` from `topk` at the first
iteration.**
Either `total_nnz_B_rt` exceeds $6 S P$ (few shapes or joints) or
`max_influences + 1` exceeds the number of joints. Lower `total_nnz_B_rt`, or
use more joints ([Compressing](compressing.md#choosing-settings)).

**`ValueError: rest_joint_matrices must have shape (P, 4, 4)`.**
You passed $3 \times 4$ matrices, or a flat list that was not reshaped. Use
`np.array(list_of_16_floats).reshape(-1, 4, 4)`.

**`FileNotFoundError` when saving.**
`run` does not create the output directory.

**`CUDA out of memory`.**
Close other GPU processes; compress large separate meshes separately; as a
last resort run on CPU with `CUDA_VISIBLE_DEVICES=""`.

**Loss goes to `nan`.**
Almost always degenerate geometry: duplicated vertices, zero-area faces, or a
delta array containing `nan` or `inf`. Check with `np.isfinite(deltas).all()`
and inspect the mesh.

**The solve takes an hour.**
That is CPU speed. See the GPU section of [Installation](../getting_started/installation.md).

## Result quality

**Mean error is fine, maximum error is large.**
Find the shape and the vertex ([Evaluating results](evaluating_results.md)).
If it is inside the mouth or eye bag, it is probably invisible. Otherwise more
iterations, then more joints.

**Result is blurry or loses creases.**
Lower `alpha`, then raise `total_nnz_B_rt`.

**Weight map is speckled; the mesh shows small bumps on some shapes.**
Raise `alpha`.

**Two machines give different numbers.**
Expected across CPU/GPU and torch versions. Both results are valid.

## Using the output

**Mesh explodes into noise as soon as a shape is turned on.**
Vertex order in the target mesh differs from the compressed one, or the weight
columns are not in joint order.

**Every shape is slightly wrong and the error grows as shapes combine.**
Matrix convention (row versus column vectors) or the centring offset.
[Pipeline integration](pipeline_integration.md#coordinate-conventions).

**One shape produces a different expression than its name.**
Shape order does not match the names you recorded. The compressor keeps the
order of the input `deltas`; nothing else.

**Shapes look right one by one but the face doubles up in the rig.**
The original blendShape deformer is still active on the same mesh.

**Result is close but stiff on large motions such as jaw open.**
Scale and shear were discarded: joint channels not all connected, a bake that
skipped shear, or a rigid-only skinning path in the engine.

**Result was good and degraded after weights were "cleaned up".**
Weights were smoothed, pruned or renormalised after binding. Reload them from
the file.

**`generate_frames` raises `KeyError: 'num_indices'`.**
Your model file has empty rig-logic metadata, which is normal for exported
data. `generate_frames` is only for the sample rigs; drive
`compute_frame_vertices` with your own shape coefficients instead.

## Still stuck

Open an issue at
[github.com/martinlanton/meta-compskin/issues](https://github.com/martinlanton/meta-compskin/issues)
with the model summary printed by the compressor (`Model: ... deltas: ...`), the
settings you changed, the full error text, and, for quality problems, the
`maxDelta` and `meanDelta` lines.
