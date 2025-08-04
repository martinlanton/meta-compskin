# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import igl
import numpy as np
import scipy as sp
import torch
import time
from pathlib import Path

import metacompskin.rig.riglogic as rl


location = Path(__file__).parent
data_location = location / "data"
out_location = data_location / "out"


class SkinCompressor:
    def __init__(self, model_file=None, iterations=10000):
        self.iterations = iterations
        self.model = "aura"
        self.number_of_bones = 40
        self.max_influences = 8  # number of weights per vertex
        self.total_nnz_Brt = 6000  # number of non-zero values into Brt matrix
        self.init_weight = 1e-3
        self.power = 2

        # TODO : The alpha is used for Laplacian regularization. Lower values lead to sharper
        #  results, whereas higher values lead to smoother results. That means that alpha values
        #  should be lower for high density meshes and higher for low density meshes.
        #  As a result, that means we can probably proceduralize the computation of the alpha value
        #  based on the mesh density (number of points per size).
        # Aura : 5944
        self.alpha_values = {"aura": 10, "jupiter": 10, "proteus": 50, "bowen": 50}
        self.alpha = self.alpha_values[self.model]

        self.seed = 12345
        torch.manual_seed(self.seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.npb = self.get_model(model_file)

        self.n_bs = self.npb["deltas"].shape[0]
        self.rest_verts = self.npb["rest_verts"]
        print(f"number of vertices: {self.rest_verts.shape[0]}")
        self.rest_faces = self.npb["rest_faces"]

        self.loss_list = []
        self.abserr_list = []

    def get_model(self, model_file):
        """Loads the model data from a file or uses a default model.

        :param model_file: Path to the model file. If None, uses the default model.
        :return:
        """
        default_model = self.model

        if model_file is None:
            model_file = str(data_location / f"in/{default_model}.npz")
        return np.load(model_file, allow_pickle=True)

    def run(self, output_filename="result", generate_frames=False):
        A = self.get_matrix_for_optimization()
        N = A.shape[1]  # number of vertices

        self.L = self.get_laplacian_regularization(self.rest_faces)

        TR = self.buildTR()

        # Brt - 6 degree of freedom  per blendshape per bone  (6, numBlendshapes, numBones, 1, 1)
        Brt = (
            (self.init_weight * torch.randn((6, self.n_bs, self.number_of_bones, 1, 1)))
            .clone()
            .float()
            .to(self.device)
            .requires_grad_()
        )

        rest_centered = self.rest_verts - self.rest_verts.mean(axis=0)
        # rest_pose  nx4 aray of vertices (with forth column 1)
        self.rest_pose = (
            torch.from_numpy(add_homog_coordinate(rest_centered, 1))
            .float()
            .to(self.device)
        )
        # W PxN (numBones x numVertices) weights one per vertex per bone
        W = (
            (1e-8 * torch.randn(self.number_of_bones, N))
            .clone()
            .float()
            .to(self.device)
            .requires_grad_()
        )

        self.train(Brt=Brt, TR=TR, A=A, W=W, normalizeW=False)
        self.train(Brt=Brt, TR=TR, A=A, W=W, normalizeW=True)

        Wn = W / W.sum(axis=0)
        print(Wn.min().item(), Wn.max().item())
        BX, _, _ = self.compBX(Wn, Brt, TR, self.n_bs, self.number_of_bones)
        orig_deltas = npf(A.transpose(1, 0).reshape(-1, self.n_bs, 3))
        our_deltas = npf(BX.transpose(1, 0).reshape(-1, self.n_bs, 3))

        maxDelta = np.abs(orig_deltas - our_deltas).max()
        meanDelta = np.abs(orig_deltas - our_deltas).mean()
        print(f"maxDelta {maxDelta}")
        print(f"meanDelta {meanDelta}")

        _, B, _ = self.compBX(Wn, Brt, TR, self.n_bs, self.number_of_bones)

        shapeXforms = B.detach().cpu().numpy()

        file_to_save = out_location / f"{output_filename}.npz"

        np.savez(
            file_to_save,
            rest=npf(self.rest_pose[:, :3]),
            quads=self.rest_faces,
            weights=npf(Wn).transpose(),
            restXform=np.array([np.eye(3, 4)] * self.number_of_bones),
            shapeXform=shapeXforms,
        )

        # TODO : determine if we really need to generate frames or if those are purely for visualization of the result
        # TODO : extract this altogether to a separate method
        if generate_frames:
            self.generate_output_frames(
                Wn, self.rest_faces, self.rest_pose, shapeXforms
            )

    def get_matrix_for_optimization(self) -> torch.Tensor:
        """Returns a blendshape matrix A that is the goal for optimization.

        The matrix's shape is: (num_blendShapes * 3 (x, y, z), num vertices)"""
        deltas = (
            self.npb["deltas"].transpose(1, 0, 2).reshape(-1, self.n_bs * 3).transpose()
        )
        return torch.from_numpy(deltas).float().to(self.device)

    def get_laplacian_regularization(self, rest_faces):
        # Adjacency matrix: represents the connectivity of the mesh
        adj = igl.adjacency_matrix(rest_faces)
        # Diagonal adjacency matrix: calculates the degree of each vertex
        adj_diag = np.array(np.sum(adj, axis=1)).squeeze()
        # Rigidness Laplacian regularization.
        # ⎧-1        if k = i
        # ⎨1/|N(i)|, if k ∈ N(i)
        # ⎩0         otherwise.
        # Where N(i) denotes all the 1-ring neighbours of i
        Lg = sp.sparse.diags(1 / adj_diag) @ (adj - sp.sparse.diags(adj_diag))
        # Return the Laplacian regularization matrix as a sparse tensor
        return torch.from_numpy((Lg).todense()).float().to(self.device).to_sparse()

    def train(
        self,
        Brt: torch.Tensor,
        TR: torch.Tensor,
        A: torch.Tensor,
        W: torch.Tensor,
        normalizeW=False,
    ):
        param_list = [Brt, W]
        self.optimizer = torch.optim.Adam(param_list, lr=1e-3, betas=(0.9, 0.9))

        st = time.time()
        for i in range(self.iterations):
            if normalizeW:
                Wn = W / W.sum(axis=0)
            else:
                Wn = W

            BX, _, _ = self.compBX(Wn, Brt, TR, self.n_bs, self.number_of_bones)
            weighed_error = BX - A

            loss = weighed_error.pow(self.power).mean().pow(2 / self.power)
            if self.alpha is not None:
                # add Laplacian regularization term
                loss += self.alpha * (self.L @ (BX).transpose(0, 1)).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                Wcutoff = torch.topk(W, self.max_influences + 1, dim=0).values[-1, :]
                Wmask = W > Wcutoff
                Wpruned = Wmask * W
                W.copy_(Wpruned)
                W.clamp_(min=0)

                Bdecider = Brt.abs()
                Bcutoff = torch.topk(Bdecider.flatten(), self.total_nnz_Brt).values[-1]
                Bmask = Bdecider >= Bcutoff
                Bpruned = Bmask * Brt
                Brt.copy_(Bpruned)

            if i % 200 == 0:
                BX, _, _ = self.compBX(Wn, Brt, TR, self.n_bs, self.number_of_bones)
                trunc_err = (BX - A).abs().max().item()

                if self.device == "cuda":
                    torch.cuda.synchronize()

                print(
                    f"{i:05d}({time.time() - st:.3f}) {loss.item():.5e} {trunc_err:.5e} {(Brt.abs() > 1e-4).count_nonzero().item()} {(W.abs() > 1e-4).count_nonzero().item()}"
                )
                self.loss_list.append(loss.item())
                self.abserr_list.append(trunc_err)

                st = time.time()

    def compBX(self, Wn, Brt, TR, n_bs, P):
        # calculates Linear Blend Skinning
        # Wn ∈ PxN   (numBones x numVertices)
        # Brt - 6 degree of freedom  per blendshape per bone  (6, n_bs, numBones, 1, 1)
        # TR  - 6 base matrices (n_bx, numBones, 3, 4)[6]  one per degree of freedom these are used to convert Brt to B
        # rest_pose  ∈ nx4
        # X :  rest_pose.p * weight
        #         vertex...vertex
        #            0      N
        #         ┌           ┐
        # bone0  x│┌───┐      │
        #        y││  →│...   │
        #        z││w*p│      │
        #        w│└───┘      │
        # bone1  x│           │
        #        y│           │
        #        z│           │
        #        w│           │
        #         ┆           ┆
        # boneP  w│           │
        #         └           ┘
        X = (Wn.unsqueeze(2) * self.rest_pose).permute(0, 2, 1).reshape(4 * P, -1)
        B = Brt[0, ...] * TR[0]
        for i in range(1, 6):
            B += Brt[i, ...] * TR[i]
        B = B.permute(0, 2, 1, 3).reshape(n_bs * 3, P * 4)
        # B current bone transforms
        #               bone 0... bone N
        #                0123     0123
        #              ┌               ┐
        # blendshape0 0│┌────┐   ┌────┐│
        #             1││TM  │...│TM  ││
        #             2│└────┘   └────┘│
        # blendshape1 0│┌────┐   ┌────┐│
        #             1││TM  │...│TM  ││
        #             2│└────┘   └────┘│
        #              │               │
        #              ┆               ┆
        #              └               ┘
        return B @ X, B, X

    def generate_output_frames(self, Wn, quads, rest_pose, shapeXforms):
        inbetween_dict = self.npb["inbetween_info"].item()
        corrective_dict = self.npb["combination_info"].item()
        test_anim_file = data_location / "in/test_anim.npz"
        test_anim = np.load(test_anim_file)
        # anim_weights num_frames x num_blendshapes
        # one weight per blendshape per frame
        anim_weights = rl.compute_rig_logic(
            torch.from_numpy(test_anim["weights"][:, :72]).float(),
            inbetween_dict,
            corrective_dict,
        ).numpy()
        num_frames = anim_weights.shape[0]
        print(num_frames, anim_weights.shape[1])
        for i in range(num_frames):
            T = generateXforms(anim_weights[i, :], shapeXforms)
            X = npf(
                (Wn.unsqueeze(2) * rest_pose)
                .permute(0, 2, 1)
                .reshape(4 * self.number_of_bones, -1)
            )
            anim_verts = T @ X
            filename = data_location / f"out/anim_frame{i:05d}.obj"
            igl.write_obj(str(filename), anim_verts.transpose(), quads)

    def buildTR(self):
        # fmt: off
        ebase = torch.tensor([[[0, 0,  0, 0],
                               [0, 0, -1, 0],
                               [0, 1,  0, 0]],

                              [[ 0, 0, 1, 0],
                               [ 0, 0, 0, 0],
                               [-1, 0, 0, 0]],

                              [[0, -1, 0, 0],
                               [1,  0, 0, 0],
                               [0,  0, 0, 0]],

                              [[0, 0, 0, 1],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]],

                              [[0, 0, 0, 0],
                               [0, 0, 0, 1],
                               [0, 0, 0, 0]],

                              [[0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 1]]], dtype=torch.float32).to(self.device)
        return ebase.reshape(6, 1, 1, 3, 4)
        # fmt: on


def add_homog_coordinate(M, dim):
    x = list(M.shape)
    x[dim] = 1
    return np.concatenate([M, np.ones(x)], axis=dim).astype(M.dtype)


def generateXforms(weights, shapeXforms):
    # weights ... (num_shapes, 1), output of riglogic
    # shapeXforms ... (3*num_shapes, 4*num_proxy_bones) matrix
    # returns: (num_proxy_bones, 3, 4) skinning transforms, input to skinCluster

    nShapes = weights.shape[0]
    nBones = shapeXforms.shape[1] // 4
    Z = weights.reshape(1, 1, nShapes) * np.dstack([np.eye(3)] * nShapes)
    # Z:
    # ┌      ┐┌      ┐┌      ┐
    # │w₁   0││w₂   0││w₃   0│
    # │  w₁  ││  w₂  ││  w₃  │  ───▶ axis 2
    # │0   w₁││0   w₂││0   w₃│
    # └      ┘└      ┘└      ┘
    #
    # Z.transpose(0, 2, 1).reshape(3, -1)
    # ┌                  ┐
    # │w₁0 0 w₂0 0 w₃0 0 │
    # │0 w₁0 0 w₂0 0 w₃0 │
    # │0 0 w₁0 0 w₂0 0 w₃│
    # └                  ┘
    # Z.transpose(0, 2, 1).reshape(3, -1) @ shapeXforms
    #   weighted sum of blendshape transforms (3, 4 * num_bones)
    #
    # Z.transpose(0, 2, 1).reshape(3, -1) @ shapeXforms + np.array([np.eye(3, 4)] * nBones).transpose(1, 0, 2).reshape(3, -1)
    #   add 1 to diagonals for every transform (befor was 0)
    res = Z.transpose(0, 2, 1).reshape(3, -1) @ shapeXforms + np.array(
        [np.eye(3, 4)] * nBones
    ).transpose(1, 0, 2).reshape(3, -1)
    return res


def npf(T):
    return T.detach().cpu().numpy()


# TODO : refactor to make it usable with parameters
# TODO : train with different 3d models?
# TODO : add tests for our output
# TODO : refactor as needed
# TODO : document
