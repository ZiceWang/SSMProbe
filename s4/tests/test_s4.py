import torch

from torch_s4 import RegularizationConfig, S4Layer, hippo_legs_matrix, regularization_loss


def test_hippo_legs_matrix_shape_and_triangle() -> None:
    A, B = hippo_legs_matrix(5)
    assert A.shape == (5, 5)
    assert B.shape == (5,)
    assert torch.allclose(torch.triu(A, diagonal=1), torch.zeros_like(torch.triu(A, diagonal=1)))


def test_forward_and_step_match() -> None:
    torch.manual_seed(0)
    layer = S4Layer(d_model=4, state_dim=8, dropout=0.0)
    x = torch.randn(2, 12, 4)
    y_parallel = layer(x)

    state = None
    outputs = []
    for t in range(x.shape[1]):
        y_t, state = layer.step(x[:, t], state)
        outputs.append(y_t.unsqueeze(1))
    y_stream = torch.cat(outputs, dim=1)

    assert y_parallel.shape == y_stream.shape
    assert torch.allclose(y_parallel, y_stream, atol=1e-4, rtol=1e-4)


def test_regularization_is_finite() -> None:
    layer = S4Layer(d_model=3, state_dim=4)
    loss = regularization_loss(layer, RegularizationConfig(parameter_decay_weight=1.0))
    assert torch.isfinite(loss)


def test_train_loop_smoke() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    layer = S4Layer(d_model=4, state_dim=8, dropout=0.0).to(device)
    head = torch.nn.Linear(4, 1).to(device)
    optimizer = torch.optim.Adam(list(layer.parameters()) + list(head.parameters()), lr=1e-2)

    inputs = torch.randn(8, 16, 4, device=device)
    targets = torch.randn(8, 16, 1, device=device)

    initial_loss = None
    final_loss = None

    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        outputs = head(layer(inputs))
        loss = torch.nn.functional.mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        current_loss = float(loss.detach().cpu())
        if initial_loss is None:
            initial_loss = current_loss
        final_loss = current_loss

    assert initial_loss is not None
    assert final_loss is not None
    assert torch.isfinite(torch.tensor(initial_loss))
    assert torch.isfinite(torch.tensor(final_loss))
    assert final_loss <= initial_loss * 1.5
