from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

def plot_delaunay(ax, V_frame, title):
    tri = Delaunay(V_frame)
    ax.triplot(V_frame[:, 0],V_frame[:, 1],tri.simplices,lw=1)
    ax.scatter(V_frame[:, 0],V_frame[:, 1],s=80)

    for i, (x, y) in enumerate(V_frame):
        ax.text(x + 0.8, y + 0.8, str(i+1), fontsize=10)

    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)

    ax.set_xlim(-60, 60)
    ax.set_ylim(-35, 35)

    ax.set_aspect("equal")
    ax.set_title(title)

def plot_adjacency_cmap(ax, A, title, cmap="Greys"):
    im = ax.imshow(
        A,
        vmin=0,
        vmax=1,
        cmap=cmap,
        interpolation="nearest"
    )
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_method_cmap(ax, methods,M):
        
    im = ax.imshow(M.values, vmin=0, vmax=1, cmap="inferno")

    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_yticklabels(methods)

    # annotate values
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(
                j, i,
                f"{M.values[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if M.values[i, j] > 0.6 else "black",
                fontsize=9
            )

    ax.set_title("Pairwise agreement (F1) between formation CPD methods")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("F1 score")

def plot_n_segments(results, xlabel, title):
    """
    results: dict[param_value -> out]
    """
    xs = list(results.keys())
    ys = [results[x]["n_segments"] for x in xs]

    plt.figure(figsize=(5, 3))
    plt.plot(xs, ys, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("# formation segments")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

def plot_cp_timeline(results, xlabel, title):
    """
    Each row = one parameter value
    """
    plt.figure(figsize=(8, 4))

    for y, (param, out) in enumerate(results.items()):
        cps = out["change_points"]
        plt.vlines(
            cps,
            y - 0.4,
            y + 0.4,
            linewidth=2,
        )

    plt.yticks(
        range(len(results)),
        [str(k) for k in results.keys()],
    )
    plt.xlabel("time")
    plt.ylabel(xlabel)
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

def plot_method_cp_timeline(results, title):
    plt.figure(figsize=(8, 4))

    for y, (method, out) in enumerate(results.items()):
        cps = out["change_points"]
        plt.vlines(cps, y - 0.4, y + 0.4, linewidth=2)

    plt.yticks(
        range(len(results)),
        list(results.keys()),
    )
    plt.xlabel("time")
    plt.ylabel("CPD method")
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
