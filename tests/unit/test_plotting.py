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


def test_plot_bifurcation_marks_the_critical_ambition() -> None:
    import numpy as np

    from corruption.plotting import plot_bifurcation

    base = BearerParams(W=0.6, A=0.5)
    fig = plot_bifurcation(base, np.linspace(0, 1.5, 50), [0.0, 3.0])
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert any(label.startswith("A_c") for label in labels)
    assert sum(label.startswith("C_u") for label in labels) == 2
    plt.close(fig)


def test_plot_fate_map_has_colorbar_and_boundary() -> None:
    import numpy as np

    from corruption.bifurcation import sweep
    from corruption.plotting import plot_fate_map

    base = BearerParams(W=0.6, A=0.5)
    result = sweep(base, np.linspace(0, 1.5, 30), np.linspace(0, 6, 20))
    fig = plot_fate_map(result, base)
    assert len(fig.axes) == 2
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert "fate boundary A_f(β)" in labels
    plt.close(fig)


def test_stochastic_plots_build() -> None:
    import numpy as np

    from corruption.plotting import plot_fall_time_distribution, plot_sample_paths
    from corruption.stochastic import MonteCarloResult

    t = np.linspace(0, 10, 11)
    fig = plot_sample_paths(t, np.zeros((11, 3)), c_star=0.9)
    assert len(fig.axes[0].get_lines()) == 4
    plt.close(fig)
    result = MonteCarloResult(fall_time=np.array([1.0, 2.0, np.inf]), horizon=10.0)
    fig = plot_fall_time_distribution(result)
    assert len(fig.axes) == 2
    plt.close(fig)
    empty = MonteCarloResult(fall_time=np.array([np.inf]), horizon=10.0)
    plt.close(plot_fall_time_distribution(empty))


def test_plot_survival_curves_draws_one_curve_per_race() -> None:
    from corruption.plotting import plot_survival_curves
    from corruption.races import kaplan_meier, survival_table
    from corruption.stochastic import SDEConfig

    table = survival_table(n_per_race=50, base=SDEConfig(dt=0.2, t_max=50.0), seed=1)
    fig = plot_survival_curves(kaplan_meier(table))
    labels = {line.get_label() for line in fig.axes[0].get_lines()}
    assert {"Men", "Dwarves", "Elves"} <= labels
    plt.close(fig)
