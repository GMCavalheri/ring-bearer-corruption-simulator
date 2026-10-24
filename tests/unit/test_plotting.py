import matplotlib.pyplot as plt
import pytest

from corruption.model import BearerParams, simulate
from corruption.plotting import plot_trajectories


def test_plot_trajectories_draws_one_line_per_bearer_plus_threshold() -> None:
    trajs = [simulate(BearerParams(W=w, A=0.8), t_max=100.0) for w in (0.4, 0.95)]
    fig = plot_trajectories(trajs, ["a", "b"], c_star=0.9)
    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.get_lines()]
    assert "a" in labels and "b" in labels and "C* (fall)" in labels
    plt.close(fig)


def test_plot_trajectories_rejects_mismatched_labels() -> None:
    traj = simulate(BearerParams(W=0.5, A=0.8), t_max=10.0)
    with pytest.raises(ValueError):
        plot_trajectories([traj], ["a", "b"], c_star=0.9)
