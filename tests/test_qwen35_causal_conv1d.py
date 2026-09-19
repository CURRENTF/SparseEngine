import pytest
import torch
import torch.nn.functional as F

from sparseengine.kernels.triton.qwen3_5.causal_conv1d import causal_conv1d_fn


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


def _reference(
    x,
    weight,
    bias,
    query_start_loc,
    cache_indices,
    has_initial_state,
    conv_states,
    activation,
):
    output = torch.empty_like(x)
    expected_states = conv_states.clone()
    starts = query_start_loc.cpu().tolist()
    for sequence_id, (start, end) in enumerate(zip(starts, starts[1:])):
        state_index = int(cache_indices[sequence_id])
        sequence = x[:, start:end].unsqueeze(0)
        if bool(has_initial_state[sequence_id]):
            conv_input = torch.cat(
                [expected_states[state_index].unsqueeze(0), sequence],
                dim=-1,
            )
        else:
            conv_input = F.pad(sequence, (3, 0))
        result = F.conv1d(
            conv_input,
            weight.unsqueeze(1),
            bias,
            groups=int(weight.shape[0]),
        )
        if activation in ("silu", "swish"):
            result = F.silu(result)
        output[:, start:end] = result.squeeze(0).to(output.dtype)
        expected_states[state_index].copy_(conv_input[0, :, -3:])
    return output, expected_states


def _inputs(dim, sequence_lengths, *, transposed_layout=False):
    torch.manual_seed(dim + sum(sequence_lengths))
    device = torch.device("cuda")
    total_tokens = sum(sequence_lengths)
    if transposed_layout:
        x = torch.randn(
            total_tokens,
            dim,
            device=device,
            dtype=torch.bfloat16,
        ).T
    else:
        x = torch.randn(
            dim,
            total_tokens,
            device=device,
            dtype=torch.bfloat16,
        )
    weight = (
        torch.randn(dim, 4, device=device, dtype=torch.bfloat16) * 0.1
    )
    bias = torch.randn(dim, device=device, dtype=torch.bfloat16) * 0.1
    query_start_loc = torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    cache_indices = torch.tensor(
        list(reversed(range(len(sequence_lengths)))),
        device=device,
        dtype=torch.int32,
    )
    has_initial_state = torch.tensor(
        [index % 2 == 0 for index in range(len(sequence_lengths))],
        device=device,
        dtype=torch.bool,
    )
    conv_states = torch.randn(
        len(sequence_lengths) + 2,
        dim,
        3,
        device=device,
        dtype=torch.bfloat16,
    )
    return (
        x,
        weight,
        bias,
        query_start_loc,
        cache_indices,
        has_initial_state,
        conv_states,
    )


@pytest.mark.parametrize("transposed_layout", [False, True])
@pytest.mark.parametrize("activation", [None, "silu"])
def test_varlen_causal_conv1d_matches_torch_reference(
    transposed_layout,
    activation,
):
    tensors = _inputs(
        127,
        [1, 2, 5, 33, 65],
        transposed_layout=transposed_layout,
    )
    x, weight, bias, starts, indices, has_initial, states = tensors
    expected, expected_states = _reference(
        x,
        weight,
        bias,
        starts,
        indices,
        has_initial,
        states,
        activation,
    )

    actual = causal_conv1d_fn(
        x,
        weight,
        bias=bias,
        query_start_loc=starts,
        cache_indices=indices,
        has_initial_state=has_initial,
        conv_states=states,
        activation=activation,
    )

    torch.testing.assert_close(actual, expected, rtol=1.0e-2, atol=5.0e-2)
    torch.testing.assert_close(
        states,
        expected_states,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("dim", [5_120, 10_240])
def test_qwen36_tp_dimensions_match_torch_reference(dim):
    tensors = _inputs(dim, [1, 7])
    x, weight, _, starts, indices, has_initial, states = tensors
    expected, expected_states = _reference(
        x,
        weight,
        None,
        starts,
        indices,
        has_initial,
        states,
        "silu",
    )

    actual = causal_conv1d_fn(
        x,
        weight,
        bias=None,
        query_start_loc=starts,
        cache_indices=indices,
        has_initial_state=has_initial,
        conv_states=states,
        activation="silu",
    )

    torch.testing.assert_close(actual, expected, rtol=1.0e-2, atol=5.0e-2)
    assert torch.equal(states, expected_states)


def test_long_prefill_matches_torch_reference():
    tensors = _inputs(256, [2_049], transposed_layout=True)
    x, weight, bias, starts, indices, has_initial, states = tensors
    expected, expected_states = _reference(
        x,
        weight,
        bias,
        starts,
        indices,
        has_initial,
        states,
        "silu",
    )

    actual = causal_conv1d_fn(
        x,
        weight,
        bias=bias,
        query_start_loc=starts,
        cache_indices=indices,
        has_initial_state=has_initial,
        conv_states=states,
        activation="silu",
    )

    torch.testing.assert_close(actual, expected, rtol=1.0e-2, atol=5.0e-2)
    assert torch.equal(states, expected_states)


def test_padded_sequence_is_unchanged():
    tensors = _inputs(128, [3, 5])
    x, weight, bias, starts, indices, has_initial, states = tensors
    indices[0] = -1
    original_states = states.clone()

    actual = causal_conv1d_fn(
        x,
        weight,
        bias=bias,
        query_start_loc=starts,
        cache_indices=indices,
        has_initial_state=has_initial,
        conv_states=states,
        activation="silu",
        pad_slot_id=-1,
    )

    assert torch.equal(actual[:, :3], x[:, :3])
    assert torch.equal(states[1:], original_states[1:])


def test_causal_conv1d_rejects_unsupported_width():
    x = torch.zeros(8, 3, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"shape \[dim, 4\]"):
        causal_conv1d_fn(
            x,
            torch.zeros(8, 3, device="cuda", dtype=torch.bfloat16),
            query_start_loc=torch.tensor([0, 3], device="cuda"),
            conv_states=torch.zeros(
                1,
                8,
                3,
                device="cuda",
                dtype=torch.bfloat16,
            ),
        )
