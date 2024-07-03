# pylint: skip-file

import opt_einsum
import pytest
import torch
from accelerate.utils import set_seed
from opt_einsum import DynamicProgramming


def test_query_gradient_svd(
    seed: int = 0,
) -> None:
    input_dim = 2048
    output_dim = 1024
    batch_dim = 16
    set_seed(seed)

    gradient = torch.rand(size=(batch_dim, output_dim, input_dim), dtype=torch.float64)

    U, S, V = torch.linalg.svd(
        gradient.contiguous(),
        full_matrices=False,
    )
    assert torch.allclose(gradient, U @ torch.diag_embed(S) @ V, atol=1e-5, rtol=1e-3)

    rank = 32
    U_k = U[:, :, :rank]
    S_k = S[:, :rank]
    V_k = V[:, :rank, :].clone()
    left, right = torch.matmul(U_k, torch.diag_embed(S_k)).contiguous(), V_k.contiguous()
    assert torch.bmm(left, right).shape == gradient.shape

    rank = input_dim
    U, S, V = torch.linalg.svd(
        gradient.contiguous(),
        full_matrices=False,
    )
    U_k = U[:, :, :rank]
    S_k = S[:, :rank]
    V_k = V[:, :rank, :].clone()
    left, right = torch.matmul(U_k, torch.diag_embed(S_k)).contiguous(), V_k.contiguous()
    assert torch.allclose(torch.bmm(left, right), gradient, atol=1e-5, rtol=1e-3)

    rank = 32
    lr_gradient1 = torch.rand(size=(batch_dim, output_dim, rank), dtype=torch.float64)
    lr_gradient2 = torch.rand(size=(batch_dim, rank, input_dim), dtype=torch.float64)
    gradient = torch.bmm(lr_gradient1, lr_gradient2)
    U, S, V = torch.linalg.svd(
        gradient.contiguous(),
        full_matrices=False,
    )
    U_k = U[:, :, :rank]
    S_k = S[:, :rank]
    V_k = V[:, :rank, :].clone()
    left_mat, right_mat = torch.matmul(U_k, torch.diag_embed(S_k)).contiguous(), V_k.contiguous()
    assert torch.allclose(torch.bmm(left_mat, right_mat), gradient, atol=1e-5, rtol=1e-3)

    query_batch_dim = 32
    new_gradient = torch.rand(size=(query_batch_dim, output_dim, input_dim), dtype=torch.float64)
    score = opt_einsum.contract("toi,qoi->tq", gradient, new_gradient)

    lr_score = opt_einsum.contract("qki,toi,qok->qt", right_mat, new_gradient, left_mat)
    assert torch.allclose(score, lr_score)

    lr_score_reconst_matmul = torch.matmul(
        torch.matmul(left_mat, right_mat).view(left_mat.size(0), -1), new_gradient.view(new_gradient.shape[0], -1).t()
    )
    assert torch.allclose(score, lr_score_reconst_matmul)

    # These should be able to avoid explicit reconstruction.
    # This should be used when input_dim > output_dim.
    intermediate = opt_einsum.contract("qki,toi->qtko", right_mat, new_gradient)
    final = opt_einsum.contract("qtko,qok->qt", intermediate, left_mat)
    assert torch.allclose(score, final)
    print("Option 1")
    print(intermediate.numel())

    # This should be used when output_dim > input_dim.
    intermediate2 = torch.einsum("toi,qok->qtik", new_gradient, left_mat)
    final2 = opt_einsum.contract("qki,qtik->qt", right_mat, intermediate2)
    assert torch.allclose(score, final2)
    print("Option 2")
    print(intermediate2.numel())

    print("Reconstruction")
    print((torch.matmul(left_mat, right_mat).view(left_mat.size(0), -1)).numel())
    path = opt_einsum.contract_path("qki,toi,qok->qt", right_mat, new_gradient, left_mat)
    print(path)


@pytest.mark.parametrize("input_dim", [256, 512])
@pytest.mark.parametrize("output_dim", [512, 1024])
@pytest.mark.parametrize("batch_dim", [8, 16])
@pytest.mark.parametrize("qbatch_dim", [8, 16])
@pytest.mark.parametrize("rank", [8, 32])
@pytest.mark.parametrize("seed", [0])
def test_query_gradient_svd_reconst(
    input_dim: int,
    output_dim: int,
    batch_dim: int,
    qbatch_dim: int,
    rank: int,
    seed: int,
) -> None:
    set_seed(seed)

    lr_gradient1 = torch.rand(size=(batch_dim, output_dim, rank + 50), dtype=torch.float64)
    lr_gradient2 = torch.rand(size=(batch_dim, rank + 50, input_dim), dtype=torch.float64)
    gradient = torch.bmm(lr_gradient1, lr_gradient2)
    U, S, V = torch.linalg.svd(
        gradient.contiguous(),
        full_matrices=False,
    )
    U_k = U[:, :, :rank]
    S_k = S[:, :rank]
    V_k = V[:, :rank, :].clone()
    left_mat, right_mat = torch.matmul(U_k, torch.diag_embed(S_k)).contiguous(), V_k.contiguous()

    output_gradient = torch.rand(size=(qbatch_dim, output_dim), dtype=torch.float64)
    input_activation = torch.rand(size=(qbatch_dim, input_dim), dtype=torch.float64)
    new_gradient = opt_einsum.contract("b...i,b...o->bio", output_gradient, input_activation)

    lr_score = opt_einsum.contract("qki,toi,qok->qt", right_mat, new_gradient, left_mat)
    lr_score_reconst_matmul = torch.matmul(
        torch.matmul(left_mat, right_mat).view(left_mat.size(0), -1), new_gradient.view(new_gradient.shape[0], -1).t()
    )
    assert torch.allclose(lr_score, lr_score_reconst_matmul)

    # This should be used when input_dim > output_dim.
    intermediate = opt_einsum.contract("qki,toi->qtko", right_mat, new_gradient)
    final = opt_einsum.contract("qtko,qok->qt", intermediate, left_mat)
    assert torch.allclose(lr_score, final)
    print("Option 1")
    print(intermediate.numel())

    # This should be used when output_dim > input_dim.
    intermediate2 = torch.einsum("toi,qok->qtik", new_gradient, left_mat)
    final2 = opt_einsum.contract("qki,qtik->qt", right_mat, intermediate2)
    assert torch.allclose(lr_score, final2)
    print("Option 2")
    print(intermediate2.numel())

    print("Reconstruction")
    reconst_numel = (torch.matmul(left_mat, right_mat).view(left_mat.size(0), -1)).numel()
    print(reconst_numel)

    # path = opt_einsum.contract_path("qki,toi,qok->qt", right_mat, new_gradient, left_mat, optimize="greedy")
    # print(path)

    path = opt_einsum.contract_path(
        "qki,toi,qok->qt",
        right_mat,
        new_gradient,
        left_mat,
        optimize=DynamicProgramming(search_outer=True, minimize="size"),
    )
    print(path)

    path = opt_einsum.contract_path("qki,toi,qok->qt", right_mat, new_gradient, left_mat, optimize="optimal")
    print(path)

    print("Direct")
    path = opt_einsum.contract_path(
        "qik,qko,bi,bo->qb",
        left_mat,
        right_mat,
        output_gradient,
        input_activation,
        optimize=DynamicProgramming(search_outer=True, minimize="flops"),
    )
    direct_result = opt_einsum.contract(
        "qik,qko,b...i,b...o->qb",
        left_mat,
        right_mat,
        output_gradient,
        input_activation,
        optimize=DynamicProgramming(search_outer=True, minimize="flops"),
    )
    assert torch.allclose(lr_score, direct_result)
    print(path)

    # print("Direct")
    # reconst_numel = (torch.matmul(left_mat, right_mat).view(left_mat.size(0), -1)).numel()
    # print(reconst_numel)
    # path = opt_einsum.contract_path("qki,toi,qok->qt", right_mat, new_gradient, left_mat, optimize="optimal")
    # print(path)

    if new_gradient.size(0) * right_mat.size(0) * rank * min((right_mat.size(2), left_mat.size(1))) > right_mat.size(
        0
    ) * right_mat.size(2) * left_mat.size(1):
        assert intermediate2.numel() > reconst_numel and intermediate.numel() > reconst_numel
    elif output_dim >= input_dim:
        assert intermediate2.numel() <= reconst_numel
    else:
        assert intermediate.numel() <= reconst_numel


def test_query_gradient_svd_vs_low_rank_svd(
    seed: int = 0,
) -> None:
    input_dim = 2048
    output_dim = 1024
    batch_dim = 16
    set_seed(seed)

    rank = 32
    lr_gradient1 = torch.rand(size=(batch_dim, output_dim, rank), dtype=torch.float32)
    lr_gradient2 = torch.rand(size=(batch_dim, rank, input_dim), dtype=torch.float32)
    gradient = torch.bmm(lr_gradient1, lr_gradient2)

    U, S, V = torch.linalg.svd(
        gradient.contiguous(),
        full_matrices=False,
    )

    U_k = U[:, :, :rank]
    S_k = S[:, :rank]
    V_k = V[:, :rank, :].clone()
    left, right = torch.matmul(U_k, torch.diag_embed(S_k)).contiguous(), V_k.contiguous()
    assert torch.bmm(left, right).shape == gradient.shape
    print(f"Error: {(torch.bmm(left, right) - gradient).norm()}")

    new_U, new_S, new_V = torch.svd_lowrank(
        gradient.contiguous(),
        q=rank,
    )
    new_left, new_right = torch.matmul(new_U, torch.diag_embed(new_S)).contiguous(), new_V.transpose(1, 2).contiguous()
    assert torch.bmm(new_left, new_right).shape == gradient.shape
    print(f"Error: {(torch.bmm(new_left, new_right) - gradient).norm()}")
