"""RecurrentCNN / PlainCNN checks: parameter budget, shapes, single-rollout logits."""
import torch

from cf.models.rcnn import RecurrentCNN, PlainCNN, n_params


def test_recurrent_cnn_param_count_hid22_is_9989():
    # encoder 3*22*9+22 + cell (25*22*9+22 + 22*22*9+22) + readout 22+1 = 9989
    assert n_params(RecurrentCNN(hid=22)) == 9989
    assert n_params(RecurrentCNN()) == 9989       # notebook default hid=22


def test_recurrent_cnn_output_shape():
    torch.manual_seed(0)
    model = RecurrentCNN(hid=22).eval()
    x = torch.randn(2, 3, 11, 11)
    with torch.no_grad():
        out = model(x, T=8)
    assert out.shape == (2, 11, 11)
    assert model(x, T=1).shape == (2, 11, 11)


def test_logits_at_equals_forward():
    torch.manual_seed(0)
    model = RecurrentCNN(hid=22).eval()
    x = torch.randn(2, 3, 11, 11)
    Ts = [1, 4, 8]
    with torch.no_grad():
        outs = model.logits_at(x, Ts)
    assert sorted(outs.keys()) == Ts
    for T in Ts:
        with torch.no_grad():
            fwd = model(x, T=T)
        assert outs[T].shape == (2, 11, 11)
        torch.testing.assert_close(outs[T], fwd, rtol=0, atol=0,
                                   msg="logits_at must equal forward at T=%d" % T)


def test_logits_at_single_rollout_matches_separate_runs():
    # one rollout to max(Ts) gives the same readout as stopping and re-running
    torch.manual_seed(0)
    model = RecurrentCNN(hid=22).eval()
    x = torch.randn(1, 3, 11, 11)
    with torch.no_grad():
        outs = model.logits_at(x, [2, 5, 8])
        for T in (2, 5, 8):
            assert torch.equal(outs[T], torch.as_tensor(model(x, T=T)))


def test_plain_cnn_accepts_and_ignores_T():
    torch.manual_seed(0)
    model = PlainCNN(hid=18, depth=6).eval()
    x = torch.randn(2, 3, 11, 11)
    with torch.no_grad():
        a, b = model(x, T=1), model(x, T=8)
    assert a.shape == (2, 11, 11)
    assert torch.equal(a, b)
