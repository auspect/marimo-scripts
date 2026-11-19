# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.25.2",
#     "numpy>=2.0",
#     "scikit-learn>=1.9",
#     "wigglystuff==0.5.40",
# ]
# ///
#
# A minimal classification tree (Gini criterion), watched line by line with the
# wigglystuff StateViewer, then compared with scikit-learn: same tree, same probabilities.
# Open with: uvx marimo edit --sandbox notebooks/gini_tree_states.py

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup:
    from functools import partial

    import marimo as mo
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from wigglystuff import Slider2D, StateViewer

    # StateViewer writes out every local variable at every step: with abbreviated arrays,
    # recording is three times faster and the variables table stays readable.
    np.set_printoptions(threshold=8, edgeitems=2, precision=3)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/auspect/marimo-scripts/blob/main/notebooks/ML/decision_tree_interactive_widget.py)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # How does a decision tree choose its thresholds?

    A classification tree asks a series of questions such as "x₁ ≤ 0.42?". At each node it
    tries every possible threshold, one by one, and keeps the one that best separates the
    classes according to the Gini index.

    Here the `grow` function is run once by `StateViewer`, which records every line it
    executes and redraws the state at each step. The function contains no display code.
    Further down, scikit-learn builds the same tree, and the prediction for one point is
    followed from the root to its leaf.

    **Request**: visualise, iteration by iteration, a minimal classification tree written by hand.
    **Data**: synthetic, generated here with a fixed seed.
    **Assumptions**: two features, axis-parallel splits, tree built branch by branch.
    """)
    return


@app.cell(hide_code=True)
def _():
    dataset = mo.ui.dropdown(
        {
            "Two moons": "moons",
            "Three blobs": "blobs",
            "Checkerboard": "checkerboard",
        },
        value="Two moons",
        label="Data set",
    )
    n_points = mo.ui.slider(
        30, 80, step=10, value=50, label="Number of points", show_value=True
    )
    noise = mo.ui.slider(
        0.0, 0.4, step=0.05, value=0.15, label="Noise", show_value=True
    )
    max_depth = mo.ui.slider(
        1, 4, value=3, label="Maximum depth", show_value=True
    )
    min_leaf = mo.ui.slider(
        1, 10, value=2, label="Minimum leaf size", show_value=True
    )
    seed = mo.ui.number(0, 99, value=7, label="Seed")
    only_changes = mo.ui.switch(
        label="Skip the lines that do not change the drawing"
    )
    mo.vstack(
        [
            mo.hstack([dataset, n_points, noise], justify="start", gap=2),
            mo.hstack([max_depth, min_leaf, seed], justify="start", gap=2),
            only_changes,
        ]
    )
    return dataset, max_depth, min_leaf, n_points, noise, only_changes, seed


@app.cell(expand_output=True)
def _(X, max_depth, min_leaf, only_changes, y):
    tree = []  # filled by grow while it is recorded
    viewer = mo.ui.anywidget(
        StateViewer(
            grow,
            view,
            args=(
                X,
                y,
                (max_depth.value, min_leaf.value),
                np.arange(len(y)),
                tree,
            ),
            variables=["path", "feature", "threshold", "score", "best"],
            timeline_label="call depth",
            interval_ms=30,
            sync_throttle_ms=250,
            only_changes=only_changes.value,
        )
    )
    viewer
    return tree, viewer


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **What to look at**

    - On the right, the highlighted line is the one about to run. The table below it
      follows the threshold being tested, its score and the node's best threshold.
    - In the scatter plot, the line is the threshold being tested. Each side takes the
      colour of its dominant class, stronger when the side is purer.
    - Below the plot and to its right, every tested threshold leaves a point. The further
      the point is from the plot, the lower the weighted Gini. The green circle tracks the best threshold.
    - The tree grows at each call to `grow`. A branch's width counts the points; its
      colour goes from grey to a class colour as the node becomes pure. A branch going
      left answers "yes" to the question.
    - The timeline at the bottom plots the call depth: each rise is a recursive call.
      Click it to jump to a node; ← → move one line, Space plays.
    - The "Checkerboard" data set shows the limit of the criterion: no single threshold
      separates the classes, and the first split gains almost nothing.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The function being watched

    A node is a dictionary: the indices of its points (`rows`) and its path from the root
    (`path`, a string of `L` for "yes" and `R` for "no"). The function calls itself for
    each half and stores every node in `tree`.
    """)
    return


@app.function
def grow(X, y, limits, rows, tree, path=""):
    max_depth, min_leaf = limits
    node = {"rows": rows, "path": path}
    tree.append(node)
    x, labels = X[rows], y[rows]
    best, tested = None, []
    if len(path) < max_depth and gini(labels) > 0:
        for feature, threshold in candidates(x, min_leaf):
            left = labels[x[:, feature] <= threshold]
            right = labels[x[:, feature] > threshold]
            score = weighted_gini(left, right)
            tested.append((feature, threshold, score))
            if best is None or score < best[2]:
                best = (feature, threshold, score)
    if best is None or gini(labels) - best[2] < 1e-12:
        node["leaf"] = majority(labels)
        return
    node["split"] = best
    goes_left = x[:, best[0]] <= best[1]
    grow(X, y, limits, rows[goes_left], tree, path + "L")
    grow(X, y, limits, rows[~goes_left], tree, path + "R")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Its four helpers

    `StateViewer` only records the lines of `grow`: these functions run, but leave no
    steps. The Gini index of a group is $G = 1 - \sum_k p_k^2$, where $p_k$ is the share
    of class $k$: 0 when the group is pure, larger when it is mixed. A threshold's score
    is the average of the Gini of its two sides, weighted by their size.
    """)
    return


@app.function
def gini(labels):
    """Gini index of a group: 1 − Σ pₖ²."""
    shares = np.bincount(labels) / len(labels)
    return float(1 - (shares**2).sum())


@app.function
def weighted_gini(left, right):
    """Score of a threshold: average of the Gini of both sides, weighted by their size."""
    return (len(left) * gini(left) + len(right) * gini(right)) / (
        len(left) + len(right)
    )


@app.function
def candidates(x, min_leaf):
    """Thresholds to try: for each feature, the midpoint of each pair of neighbouring values.

    A threshold is kept only if it leaves at least `min_leaf` points on each side.
    Midpoints are rounded to three decimals, which assumes points rounded to two.
    """
    pairs = []
    for feature in range(x.shape[1]):
        values = np.sort(x[:, feature])
        n_left = np.arange(
            1, len(values)
        )  # points on the left if we cut between two neighbours
        allowed = (
            (values[:-1] < values[1:])
            & (n_left >= min_leaf)
            & (len(values) - n_left >= min_leaf)
        )
        middles = np.round((values[:-1] + values[1:]) / 2, 3)
        pairs += [(feature, float(middle)) for middle in middles[allowed]]
    return pairs


@app.function
def majority(labels):
    """Most frequent class of a group: what a leaf predicts."""
    return int(np.bincount(labels).argmax())


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The state, seen from Python

    The viewer sends the number of the displayed step back to Python.
    `viewer.widget.state` holds the actual local variables of `grow` at that step; the
    table ranks the thresholds already tested on the node, best first.
    """)
    return


@app.cell
def _(viewer):
    state = viewer.widget.state
    ranking = sorted(state.get("tested", []), key=lambda entry: entry[2])
    mo.vstack(
        [
            mo.md(
                f"**Step {viewer.widget.step} of {viewer.widget.n_steps - 1}**: "
                f"line {viewer.widget.current['line']} of `grow`, "
                f"{'node ' + state['path'] if state['path'] else 'root node'}, "
                f"{len(ranking)} thresholds tested."
            ),
            mo.ui.table(
                [
                    {
                        "threshold": rule(feature, threshold),
                        "weighted Gini": round(score, 4),
                    }
                    for feature, threshold, score in ranking
                ],
                page_size=5,
                selection=None,
                show_download=False,
            )
            if ranking
            else mo.md("_No threshold tested on this node at this step._"),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The same tree with scikit-learn

    `DecisionTreeClassifier` follows the same algorithm. Four settings matter for it to
    build the same tree:

    - `criterion="gini"` and `splitter="best"`: the same criterion, every threshold tried;
    - `max_depth` and `min_samples_leaf`: the same limits as `grow`;
    - `min_impurity_decrease=1e-12`: by default scikit-learn splits even when the Gini does
      not decrease, whereas `grow` stops;
    - `random_state`: when two thresholds tie, scikit-learn keeps the first one it meets in
      a randomly drawn feature order. So seeds 0 to 19 are tried, and the first one that
      gives the same tree is kept.
    """)
    return


@app.cell
def _(X, max_depth, min_leaf, tree, y):
    def sklearn_nodes(model):
        """The scikit-learn nodes, keyed by their path like those of `grow`."""
        t, found = model.tree_, {}

        def walk(i, path):
            is_leaf = t.children_left[i] == -1
            found[path] = {
                "n": int(t.n_node_samples[i]),
                "gini": float(t.impurity[i]),
                "split": None,
                "score": None,
            }
            if not is_leaf:
                a, b = t.children_left[i], t.children_right[i]
                found[path]["split"] = (
                    int(t.feature[i]),
                    float(t.threshold[i]),
                )
                found[path]["score"] = float(
                    (
                        t.n_node_samples[a] * t.impurity[a]
                        + t.n_node_samples[b] * t.impurity[b]
                    )
                    / t.n_node_samples[i]
                )
                walk(a, path + "L")
                walk(b, path + "R")

        walk(0, "")
        return found

    def same_node(ours, theirs):
        if (
            ours["n"] != theirs["n"]
            or abs(ours["gini"] - theirs["gini"]) > 1e-9
        ):
            return False
        if ours["split"] is None or theirs["split"] is None:
            return ours["split"] is None and theirs["split"] is None
        return (
            ours["split"][0] == theirs["split"][0]
            and abs(ours["split"][1] - theirs["split"][1]) < 1e-6
        )

    def fit(random_state):
        return DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            max_depth=max_depth.value,
            min_samples_leaf=min_leaf.value,
            min_impurity_decrease=1e-12,
            random_state=random_state,
        ).fit(X, y)

    def different_paths(theirs):
        paths = sorted(set(our_nodes) | set(theirs))
        return [
            p
            for p in paths
            if p not in our_nodes
            or p not in theirs
            or not same_node(our_nodes[p], theirs[p])
        ]

    # claude used seed here, it usually isn't necessary because this algorithm is
    # deterministic,
    # but if we have multiple "best" split there's no real rule to decide,
    # so I kept this
    def choose_seed():
        """First seed that gives the same tree; failing that, the one that differs least."""
        chosen = None
        for candidate in range(20):
            model = fit(candidate)
            different = different_paths(sklearn_nodes(model))
            if chosen is None or len(different) < len(chosen[2]):
                chosen = (model, candidate, different)
            if not different:
                break
        return chosen

    def ties_with(theirs):
        """Nodes where both trees start from the same points but split elsewhere, with both scores."""
        found, different = [], different_paths(theirs)
        for path in different:
            mine, other = our_nodes.get(path), theirs.get(path)
            if path and path[:-1] in different:
                continue  # below a node that already differs, the points are no longer the same
            if (
                mine
                and other
                and mine["n"] == other["n"]
                and mine["split"]
                and other["split"]
            ):
                found.append(
                    (
                        path,
                        mine["split"],
                        mine["score"],
                        other["split"],
                        other["score"],
                    )
                )
        return found

    our_nodes = {
        node["path"]: {
            "n": len(node["rows"]),
            "gini": gini(y[node["rows"]]),
            "split": node["split"][:2] if "split" in node else None,
            "score": node["split"][2] if "split" in node else None,
        }
        for node in tree
    }
    sk_model, sk_seed, sk_different = choose_seed()
    sk_nodes = sklearn_nodes(sk_model)
    seed0_ties = ties_with(sklearn_nodes(fit(0)))
    sk_ties = ties_with(sk_nodes)

    _verdict = (
        f"With `random_state={sk_seed}`, scikit-learn builds **exactly the same tree**: "
        f"{len(our_nodes)} nodes, same counts, same Gini, same thresholds to within one millionth."
        if not sk_different
        else f"None of the seeds 0 to 19 gives the same tree. The closest is `random_state={sk_seed}`: "
        f"{len(sk_different)} nodes differ."
    )
    _why = "".join(
        f"\n- With `random_state=0`, at {'node ' + path if path else 'the root'}: `grow` keeps "
        f"{rule(*mine)} (weighted Gini {num(mine_score, 6)}), scikit-learn {rule(*other)} ({num(other_score, 6)})."
        for path, mine, mine_score, other, other_score in seed0_ties
    )
    mo.md(
        _verdict
        + (f"\n\nThe differences come from ties:{_why}" if _why else "")
    )
    return our_nodes, sk_different, sk_model, sk_nodes, sk_seed, sk_ties


@app.cell(hide_code=True)
def _(our_nodes, sk_nodes):
    def describe(entry):
        if entry is None:
            return "missing"
        return rule(*entry["split"]) if entry["split"] else "leaf"

    mo.ui.table(
        [
            {
                "node": path or "root",
                "points": (our_nodes.get(path) or sk_nodes[path])["n"],
                "Gini": round(
                    (our_nodes.get(path) or sk_nodes[path])["gini"], 4
                ),
                "grow": describe(our_nodes.get(path)),
                "scikit-learn": describe(sk_nodes.get(path)),
                "same": "yes"
                if describe(our_nodes.get(path))
                == describe(sk_nodes.get(path))
                else "no",
            }
            for path in list(our_nodes)
            + [p for p in sk_nodes if p not in our_nodes]
        ],
        page_size=16,
        selection=None,
        show_download=False,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Prediction, step by step

    To classify a point, the tree sends it down from the root to a leaf: at each node it
    asks the node's question and goes left if the answer is yes. The probabilities
    returned are the class shares in the leaf it reaches.

    Move the point in the square, then follow `predict_proba` line by line.
    """)
    return


@app.cell(hide_code=True)
def _(X):
    picker = mo.ui.anywidget(
        Slider2D(
            x=float(np.round(X[:, 0].mean(), 2)),
            y=float(np.round(X[:, 1].mean(), 2)),
            width=200,
            height=200,
            x_bounds=(float(X[:, 0].min()), float(X[:, 0].max())),
            y_bounds=(float(X[:, 1].min()), float(X[:, 1].max())),
        )
    )
    mo.hstack(
        [
            picker,
            mo.md(
                "**The point to classify.** The square covers the range of the data: x₁ to the right, x₂ upwards."
            ),
        ],
        justify="start",
        gap=1.5,
        align="center",
    )
    return (picker,)


@app.cell
def _(X, max_depth, picker, tree, y):
    point = np.array([picker.value["x"], picker.value["y"]])
    nodes = {node["path"]: node for node in tree}
    predictor = mo.ui.anywidget(
        StateViewer(
            predict_proba,
            partial(view_predict, X=X, max_depth=max_depth.value),
            args=(point, nodes, y, int(y.max()) + 1),
            variables=["path", "feature", "threshold", "labels"],
            timeline_func=lambda path: len(path),
            timeline_label="depth reached",
            interval_ms=500,
        )
    )
    predictor
    return nodes, point


@app.function
def predict_proba(point, nodes, y, n_classes):
    path = ""
    while "split" in nodes[path]:
        feature, threshold, _ = nodes[path]["split"]
        path += "L" if point[feature] <= threshold else "R"
    labels = y[nodes[path]["rows"]]
    return np.bincount(labels, minlength=n_classes) / len(labels)


@app.cell
def _(X, nodes, point, sk_different, sk_model, sk_seed, y):
    def compare_predictions():
        """Gap between `predict_proba` and scikit-learn: on the chosen point, then on the data and 625 random points."""
        n_classes = int(y.max()) + 1
        rng = np.random.default_rng(0)
        drawn = np.column_stack(
            [rng.uniform(X[:, k].min(), X[:, k].max(), 625) for k in (0, 1)]
        )
        everywhere = np.vstack([X, drawn])
        ours = np.array(
            [predict_proba(p, nodes, y, n_classes) for p in everywhere]
        )
        gaps = np.abs(ours - sk_model.predict_proba(everywhere)).max(axis=1)
        return (
            predict_proba(point, nodes, y, n_classes),
            sk_model.predict_proba(point.reshape(1, -1))[0],
            gaps,
        )

    proba_ours, proba_sklearn, proba_gaps = compare_predictions()
    _shown = lambda values: "[" + ", ".join(num(v, 3) for v in values) + "]"  # noqa: E731
    mo.md(
        f"For the point ({num(point[0])}, {num(point[1])}): `predict_proba` returns **{_shown(proba_ours)}**, "
        f"scikit-learn (`random_state={sk_seed}`) returns **{_shown(proba_sklearn)}**.\n\n"
        f"On the {len(X)} training points and 625 randomly drawn points, the two disagree on "
        f"**{int((proba_gaps > 1e-12).sum())} points**; the largest gap is {proba_gaps.max():.1e}."
        + (
            ""
            if not sk_different
            else " The two trees are not identical, hence these gaps."
        )
        + "\n\nA point lying exactly on a threshold may be classified differently: scikit-learn converts the "
        "data to 32-bit numbers before comparing, and its threshold differs from ours around the ninth decimal."
    )
    return proba_gaps, proba_ours


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The points
    """)
    return


@app.function
def make_points(kind, n, noise, seed):
    """Three two-dimensional data sets, rounded to two decimals."""
    rng = np.random.default_rng(seed)
    if kind == "moons":
        half = n // 2
        angle = rng.uniform(0, np.pi, n)
        upper = np.column_stack(
            [np.cos(angle[:half]), np.sin(angle[:half])]
        )
        lower = np.column_stack(
            [1 - np.cos(angle[half:]), 0.5 - np.sin(angle[half:])]
        )
        X = np.vstack([upper, lower]) + rng.normal(0, noise, (n, 2))
        y = np.repeat([0, 1], [half, n - half])
    elif kind == "blobs":
        centers = np.array([[-1.0, -0.6], [1.0, -0.6], [0.0, 0.9]])
        y = np.arange(n) % 3
        X = centers[y] + rng.normal(0, 0.25 + noise, (n, 2))
    else:  # checkerboard: the class depends on both features at once
        X = rng.uniform(-1, 1, (n, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        X = X + rng.normal(0, noise / 2, (n, 2))
    return np.round(X, 2), y


@app.cell
def _(dataset, n_points, noise, seed):
    X, y = make_points(dataset.value, n_points.value, noise.value, seed.value)
    mo.md(
        f"{len(y)} points, {y.max() + 1} classes ("
        + ", ".join(
            f"{'ABC'[label]}: {size}"
            for label, size in enumerate(np.bincount(y))
        )
        + ")."
    )
    return X, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Checks

    The best threshold of each node is recomputed another way (cumulative counts over the
    sorted points) and compared with the one `grow` kept. The scikit-learn tree must be the
    same, or differ only through ties; when it is the same, the probabilities must be too.
    """)
    return


@app.cell
def _(
    X,
    min_leaf,
    proba_gaps,
    proba_ours,
    sk_different,
    sk_ties,
    tree,
    viewer,
    y,
):
    def best_score_by_cumsum(rows):
        """Smallest weighted Gini of a node, without looping over thresholds."""
        n, n_classes, scores = len(rows), y.max() + 1, []
        for feature in range(X.shape[1]):
            order = rows[np.argsort(X[rows, feature], kind="stable")]
            values = X[order, feature]
            left = np.cumsum(np.eye(n_classes)[y[order]], axis=0)[:-1]
            right = left[-1] + np.eye(n_classes)[y[order[-1]]] - left
            n_left = np.arange(1, n)
            allowed = (
                (values[1:] > values[:-1])
                & (n_left >= min_leaf.value)
                & (n - n_left >= min_leaf.value)
            )
            gini_left = 1 - ((left / n_left[:, None]) ** 2).sum(axis=1)
            gini_right = 1 - ((right / (n - n_left)[:, None]) ** 2).sum(axis=1)
            scores.extend(
                ((n_left * gini_left + (n - n_left) * gini_right) / n)[allowed]
            )
        return min(scores)

    def check_tree():
        assert viewer.widget.error is None, (
            f"recording failed: {viewer.widget.error}"
        )
        assert not viewer.widget.truncated, "recording was cut before the end"
        leaves = [node for node in tree if "leaf" in node]
        covered = np.sort(np.concatenate([node["rows"] for node in leaves]))
        assert (covered == np.arange(len(y))).all(), (
            "the leaves do not partition the points exactly"
        )
        for node in tree:
            assert ("leaf" in node) != ("split" in node), (
                f"node '{node['path']}' is neither a leaf nor split"
            )
            if "split" in node:
                assert (
                    abs(node["split"][2] - best_score_by_cumsum(node["rows"]))
                    < 1e-9
                ), (
                    f"the threshold kept at node '{node['path']}' is not the best"
                )
        return leaves

    def check_sklearn():
        for path, _, mine_score, _, other_score in sk_ties:
            assert abs(mine_score - other_score) < 1e-9, (
                f"at node '{path}', scikit-learn finds a better threshold"
            )
        assert not sk_different or sk_ties, (
            "the trees differ and no tie explains it"
        )
        assert abs(proba_ours.sum() - 1) < 1e-12, (
            "the probabilities do not sum to 1"
        )
        if not sk_different:
            assert proba_gaps.max() < 1e-12, (
                "same tree, different probabilities on the random points"
            )

    leaves = check_tree()
    check_sklearn()
    return (leaves,)


@app.cell(hide_code=True)
def _(leaves, proba_gaps, sk_different, sk_seed, tree, viewer, y):
    _root = tree[0]
    _right = sum(int(np.bincount(y[node["rows"]]).max()) for node in leaves)
    _first = (
        f"The first split kept is **{rule(*_root['split'][:2])}**: the Gini goes from "
        f"{num(gini(y))} to {num(_root['split'][2])}."
        if "split" in _root
        else "The root was not split."
    )
    _sklearn = (
        f"With the same limits and `random_state={sk_seed}`, scikit-learn builds **the same tree**, and its "
        f"probabilities are identical on the data and on 625 randomly drawn points (largest gap {proba_gaps.max():.0e})."
        if not sk_different
        else f"scikit-learn departs from this tree on {len(sk_different)} nodes, through ties broken differently."
    )
    mo.md(f"""
    ## Short answer

    At each node, `grow` tries the midpoint of each pair of neighbouring values, on each
    feature, and keeps the threshold with the lowest weighted Gini. Here the tree has
    **{len(leaves)} leaves** for {len(tree)} nodes, and running it left
    **{viewer.widget.n_steps} steps**. {_first}

    {_sklearn}

    The tree classifies **{100 * _right / len(y):.0f}%** of the points used to build it correctly.

    **Limits**: this rate is measured on the training data and says nothing about new
    points. The criterion is greedy: it picks the best split of the moment without looking
    further ahead (see the "Checkerboard" data set). Agreement with scikit-learn is checked
    for the settings shown, not proved for all of them.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix: the drawing

    `view` and `view_predict` receive the local variables of the watched function whose
    names they carry, and return an SVG. They are called at every step; identical drawings
    are stored only once. `Drawing` holds what the two views share.
    """)
    return


@app.function(hide_code=True)
def num(value, digits=3):
    """Number with a proper minus sign."""
    return f"{value:.{digits}f}".replace("-", "−")


@app.function(hide_code=True)
def rule(feature, threshold):
    """The question a threshold asks, for example "x₁ ≤ 0.425"."""
    return f"x{'₁₂'[feature]} ≤ {num(threshold)}"


@app.class_definition(hide_code=True)
class Drawing:
    """The drawing pieces shared by both views: scatter plot, tree, composition bars."""

    colors = ["#e0821e", "#1f78c8", "#b8479a"]
    green = "#2a9d6f"
    PX, PY, PS, MG = (
        34,
        66,
        220,
        44,
    )  # scatter plot: corner and side; thickness of the margins
    MB, MR = (
        PY + PS + 16,
        PX + PS + 8,
    )  # bottom margin (x₁) and right margin (x₂)
    TX, GROUND = 424, 512  # tree: trunk axis and ground level

    def __init__(self, X, y, nodes):
        self.X, self.y, self.nodes = X, y, nodes
        self.n_total, self.n_classes = len(y), int(y.max()) + 1
        pad = 0.06 * (X.max(axis=0) - X.min(axis=0))
        self.low, self.high = X.min(axis=0) - pad, X.max(axis=0) + pad
        self.palette = np.array(
            [
                [int(c[i : i + 2], 16) for i in (1, 3, 5)]
                for c in self.colors[: self.n_classes]
            ]
        )

    def sx(self, value):
        return (
            self.PX
            + (value - self.low[0])
            / (self.high[0] - self.low[0])
            * self.PS
        )

    def sy(self, value):
        return (
            self.PY
            + self.PS
            - (value - self.low[1])
            / (self.high[1] - self.low[1])
            * self.PS
        )

    def box(self, path):
        """Rectangle [x₁ min, x₁ max, x₂ min, x₂ max] of a node, from the splits of its ancestors."""
        bounds = [self.low[0], self.high[0], self.low[1], self.high[1]]
        for i, side in enumerate(path):
            feature, value = self.nodes[path[:i]]["split"][:2]
            bounds[2 * feature + (side == "L")] = value
        return bounds

    def rect(self, b, attributes):
        return (
            f'<rect x="{self.sx(b[0]):.1f}" y="{self.sy(b[3]):.1f}" width="{self.sx(b[1]) - self.sx(b[0]):.1f}" '
            f'height="{self.sy(b[2]) - self.sy(b[3]):.1f}" {attributes}/>'
        )

    def purity(self, group):
        return 1 - gini(group) / (1 - 1 / self.n_classes)

    def tint(self, b, group):
        return self.rect(
            b,
            f'fill="{self.colors[majority(group)]}" opacity="{0.08 + 0.26 * self.purity(group):.2f}"',
        )

    def cut(self, b, feature, value, attributes):
        if feature == 0:
            ends = (
                self.sx(value),
                self.sy(b[3]),
                self.sx(value),
                self.sy(b[2]),
            )
        else:
            ends = (
                self.sx(b[0]),
                self.sy(value),
                self.sx(b[1]),
                self.sy(value),
            )
        return '<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" {}/>'.format(
            *ends, attributes
        )

    def wood(self, group):
        """Colour of a branch: grey if the node is mixed, class colour if it is pure."""
        shares = np.bincount(group, minlength=self.n_classes) / len(group)
        grey = np.array([154, 146, 137])
        mixed = (
            grey
            + (shares @ self.palette - grey) * self.purity(group) ** 0.6
        )
        return "#" + "".join(f"{int(channel):02x}" for channel in mixed)

    def start(self, said):
        """SVG opening, animations, and three lines of text."""
        return (
            '<svg width="540" height="540" viewBox="0 0 540 540" font-family="system-ui, sans-serif" '
            'font-size="10" fill="currentColor">'
            "<style>.ag-grow{stroke-dasharray:1;animation:ag-grow .6s ease-out}"
            "@keyframes ag-grow{from{stroke-dashoffset:1}to{stroke-dashoffset:0}}"
            ".ag-pop,.ag-ring{transform-box:fill-box;transform-origin:center}"
            ".ag-pop{animation:ag-pop .45s cubic-bezier(.34,1.56,.64,1)}@keyframes ag-pop{from{transform:scale(0)}}"
            ".ag-ring{animation:ag-ring 1.4s ease-in-out infinite}"
            "@keyframes ag-ring{50%{transform:scale(1.3);opacity:.3}}</style>"
            f'<text x="0" y="13" font-size="12.5" font-weight="600">{said[0]}</text>'
            f'<text x="0" y="31" font-size="11.5" opacity="0.85">{said[1]}</text>'
            f'<text x="0" y="48" font-size="11.5" opacity="0.85">{said[2]}</text>'
        )

    def frames(self, margins):
        """Scatter frame and ticks; with `margins`, the two margins where the Gini values are plotted."""
        frame = 'rx="3" fill="currentColor" fill-opacity="0.04" stroke="currentColor" stroke-opacity="0.2"'
        out = [
            f'<rect x="{self.PX}" y="{self.PY}" width="{self.PS}" height="{self.PS}" {frame}/>'
        ]
        for axis in (0, 1):
            size = 1.0 if self.high[axis] - self.low[axis] > 2.5 else 0.5
            for k in range(
                int(np.ceil(self.low[axis] / size)),
                int(np.floor(self.high[axis] / size)) + 1,
            ):
                if axis == 0:
                    out.append(
                        f'<text x="{self.sx(k * size):.0f}" y="{self.PY + self.PS + 11}" text-anchor="middle" opacity="0.55">{num(k * size, 1)}</text>'
                    )
                else:
                    out.append(
                        f'<text x="{self.PX - 4}" y="{self.sy(k * size) + 3:.0f}" text-anchor="end" opacity="0.55">{num(k * size, 1)}</text>'
                    )
        if margins:
            out.append(
                f'<rect x="{self.PX}" y="{self.MB}" width="{self.PS}" height="{self.MG}" {frame}/>'
            )
            out.append(
                f'<rect x="{self.MR}" y="{self.PY}" width="{self.MG}" height="{self.PS}" {frame}/>'
            )
            out.append(
                f'<text x="{self.PX + self.PS / 2}" y="{self.MB + self.MG + 12}" text-anchor="middle" opacity="0.7">Weighted Gini of the thresholds on x₁</text>'
            )
            out.append(
                f'<text x="{self.MR + self.MG / 2}" y="{self.PY - 5}" text-anchor="middle" opacity="0.7">on x₂</text>'
            )
        return "".join(out)

    def leaf_regions(self):
        return "".join(
            self.tint(self.box(item["path"]), self.y[item["rows"]])
            for item in self.nodes.values()
            if "leaf" in item
        )

    def points(self, rows):
        """The points, one path per colour; those outside `rows` are dimmed."""
        inside = np.zeros(self.n_total, dtype=bool)
        inside[rows] = True
        out = []
        for label in range(self.n_classes):
            for mask, opacity in ((inside, 1), (~inside, 0.18)):
                dots = "".join(
                    f"M{self.sx(a):.1f},{self.sy(b):.1f}h0"
                    for a, b in self.X[(self.y == label) & mask]
                )
                if dots:
                    out.append(
                        f'<path d="{dots}" stroke="{self.colors[label]}" stroke-width="5.5" stroke-linecap="round" opacity="{opacity}"/>'
                    )
        return "".join(out)

    def split_lines(self, current):
        """The splits already kept; the one of node `current` in green."""
        out = []
        for item in self.nodes.values():
            if "split" in item:
                style = (
                    f'stroke="{self.green}" stroke-width="2.6"'
                    if item["path"] == current
                    else 'stroke="currentColor" stroke-opacity="0.55" stroke-width="1.2"'
                )
                out.append(
                    self.cut(
                        self.box(item["path"]), *item["split"][:2], style
                    )
                )
        return "".join(out)

    def composition(self, top, name, group, whole, last):
        """One bar per group: its length counts the points, its colours the classes."""
        parts, start = (
            [
                f'<text x="30" y="{top + 8}" text-anchor="end" opacity="0.8">{name}</text>'
            ],
            38.0,
        )
        for label, size in enumerate(
            np.bincount(group, minlength=self.n_classes)
        ):
            length = 130 * size / whole
            if size:
                parts.append(
                    f'<rect x="{start:.1f}" y="{top}" width="{length:.1f}" height="9" fill="{self.colors[label]}"/>'
                )
            start += length
        parts.append(
            f'<text x="218" y="{top + 8}" text-anchor="end">{len(group)} pts</text>'
        )
        parts.append(
            f'<text x="270" y="{top + 8}" text-anchor="end" font-weight="600">{last}</text>'
        )
        return "".join(parts)

    def tree(
        self, max_depth, current=None, newborn=None, popped=None, lit=None
    ):
        """The tree. A node's place depends only on its path, so it never moves.

        `current` carries the ring, `newborn` grows, `popped` blossoms; with `lit`, only those paths stay vivid.
        """
        level = min(118, 392 / max(1, max_depth))
        trunk = 26

        def spot(path):
            shift = sum(
                (-1 if side == "L" else 1) * 104 / 2 ** (i + 1)
                for i, side in enumerate(path)
            )
            return self.TX + shift, self.GROUND - 44 - len(path) * level

        def width(item):
            return trunk * len(item["rows"]) / self.n_total

        out, marks = (
            [
                f'<ellipse cx="{self.TX}" cy="{self.GROUND + 3}" rx="84" ry="6" opacity="0.14"/>'
            ],
            [],
        )
        for item in self.nodes.values():
            path = item["path"]
            x_end, y_end = spot(path)
            group = self.y[item["rows"]]
            faded = (
                ' opacity="0.22"'
                if lit is not None and path not in lit
                else ""
            )
            if path:
                parent = self.nodes[path[:-1]]
                x_start, y_start = spot(parent["path"])
                # both branches leave side by side: their widths add up to the parent's
                x_start += (
                    (-1 if path[-1] == "L" else 1)
                    * (width(parent) - width(item))
                    / 2
                )
                bend = level * 0.55
            else:
                x_start, y_start, bend = self.TX, self.GROUND, 16
            out.append(
                f'<path d="M{x_start:.1f},{y_start:.1f}C{x_start:.1f},{y_start - bend:.1f} {x_end:.1f},{y_end + bend:.1f} '
                f'{x_end:.1f},{y_end:.1f}" fill="none" stroke="{self.wood(group)}" stroke-width="{max(1.6, width(item)):.1f}" '
                f'stroke-linecap="round"{faded}'
                + (
                    ' pathLength="1" class="ag-grow"/>'
                    if path == newborn
                    else "/>"
                )
            )
            if "leaf" in item:
                radius = 3.5 + 7 * (len(group) / self.n_total) ** 0.5
                marks.append(
                    f'<circle cx="{x_end:.1f}" cy="{y_end:.1f}" r="{radius:.1f}" fill="{self.colors[item["leaf"]]}" '
                    f'fill-opacity="{0.4 + 0.6 * self.purity(group):.2f}"{faded}'
                    + (' class="ag-pop"/>' if path == popped else "/>")
                )
                if (
                    len(path) < 4
                ):  # on the last level, the leaves are too close together for a number
                    marks.append(
                        f'<text x="{x_end:.1f}" y="{y_end - radius - 4:.1f}" text-anchor="middle" opacity="0.7">{len(group)}</text>'
                    )
            elif "split" in item and len(path) < 2:
                anchor, shift = (
                    ("end", -width(item) / 2 - 9)
                    if path == "L"
                    else ("start", width(item) / 2 + 9)
                )
                marks.append(
                    f'<text x="{x_end + shift:.1f}" y="{y_end + 4:.1f}" text-anchor="{anchor}">{rule(*item["split"][:2])}</text>'
                )
            elif "split" not in item:
                marks.append(
                    f'<circle cx="{x_end:.1f}" cy="{y_end:.1f}" r="3" fill="none" stroke="currentColor" stroke-width="1.5"/>'
                )
        out += marks
        if current in self.nodes:
            x_now, y_now = spot(current)
            out.append(
                f'<circle class="ag-ring" cx="{x_now:.1f}" cy="{y_now:.1f}" r="{max(8, width(self.nodes[current]) / 2 + 5):.1f}" '
                f'fill="none" stroke="{self.green}" stroke-width="2"/>'
            )
        for label in range(self.n_classes):
            out.append(
                f'<circle cx="{340 + 62 * label}" cy="533" r="4" fill="{self.colors[label]}"/>'
            )
            out.append(
                f'<text x="{348 + 62 * label}" y="536.5" opacity="0.8">class {"ABC"[label]}</text>'
            )
        return "".join(out)


@app.function(hide_code=True)
def view(
    X,
    y,
    limits,
    rows,
    tree,
    path,
    node=None,
    feature=None,
    threshold=None,
    best=None,
    tested=(),
):
    """Draws a state of `grow`: the scatter plot and tested threshold, the Gini values so far, the tree."""
    d = Drawing(X, y, {item["path"]: item for item in tree})
    labels = y[rows]
    node_gini = gini(labels)
    decided = node is not None and ("split" in node or "leaf" in node)
    scanning = node is not None and not decided and threshold is not None
    if scanning:
        goes_left = X[rows, feature] <= threshold
        left, right = labels[goes_left], labels[~goes_left]
        score = weighted_gini(left, right)
        is_best = bool(best) and best[:2] == (feature, threshold)

    # --- three lines of text: the node, the tested threshold, the best threshold
    said = [
        f"{'Node ' + path if path else 'Root'}: {len(rows)} points, Gini {num(node_gini)}"
    ]
    if node is not None and "leaf" in node:
        said[0] += f" → leaf of class {'ABC'[node['leaf']]}"
    if node is not None and "split" in node:
        said[0] += f" → split kept: {rule(*node['split'][:2])}"
    if scanning:
        said.append(
            f"Threshold tested: {rule(feature, threshold)}? Weighted Gini {num(score)}"
        )
        said[1] += ", the best so far" if is_best else ""
    else:
        said.append(f"Thresholds tested on this node: {len(tested)}")
    if best:
        said.append(
            f"Best threshold: {rule(*best[:2])}, weighted Gini {num(best[2])}, gain {num(node_gini - best[2])}"
        )
    else:
        said.append("Best threshold: none yet")
    out = [d.start(said), d.frames(margins=True), d.leaf_regions()]

    # --- the scatter plot: both sides of the tested threshold, the points, the splits
    here = d.box(path)
    if scanning:
        sides = (
            (
                [here[0], threshold, here[2], here[3]],
                [threshold, here[1], here[2], here[3]],
            )
            if feature == 0
            else (
                [here[0], here[1], here[2], threshold],
                [here[0], here[1], threshold, here[3]],
            )
        )
        out += [d.tint(sides[0], left), d.tint(sides[1], right)]
    out += [
        d.points(rows),
        d.split_lines(path if node is not None else None),
    ]
    if scanning:
        if best and not is_best:
            out.append(
                d.cut(
                    here,
                    *best[:2],
                    f'stroke="{d.green}" stroke-width="1.8" stroke-dasharray="4 3"',
                )
            )
        out.append(
            d.cut(
                here,
                feature,
                threshold,
                f'stroke="{d.green}" stroke-width="2.4"'
                if is_best
                else 'stroke="currentColor" stroke-width="1.8"',
            )
        )
    out.append(
        d.rect(
            here,
            'fill="none" stroke="currentColor" stroke-opacity="0.8" stroke-width="1.3"',
        )
    )

    # --- the margins: the weighted Gini of each threshold already tested on this node
    marks = list(tested)
    if scanning and (not marks or marks[-1][:2] != (feature, threshold)):
        marks.append((feature, threshold, score))
    if marks and node_gini > 0:

        def at(mark):
            away = (
                d.MG - 4 - mark[2] / node_gini * (d.MG - 10)
            )  # further from the plot = lower Gini
            return (
                (d.sx(mark[1]), d.MB + away)
                if mark[0] == 0
                else (d.MR + away, d.sy(mark[1]))
            )

        out.append(
            f'<rect x="{d.sx(here[0]):.1f}" y="{d.MB}" width="{d.sx(here[1]) - d.sx(here[0]):.1f}" height="{d.MG}" opacity="0.07"/>'
        )
        out.append(
            f'<rect x="{d.MR}" y="{d.sy(here[3]):.1f}" width="{d.MG}" height="{d.sy(here[2]) - d.sy(here[3]):.1f}" opacity="0.07"/>'
        )
        dashed = 'stroke="currentColor" stroke-opacity="0.5" stroke-dasharray="3 3"'
        out.append(
            f'<line x1="{d.PX}" x2="{d.PX + d.PS}" y1="{d.MB + 6}" y2="{d.MB + 6}" {dashed}/>'
        )
        out.append(
            f'<line x1="{d.MR + 6}" x2="{d.MR + 6}" y1="{d.PY}" y2="{d.PY + d.PS}" {dashed}/>'
        )
        out.append(
            f'<text x="{d.PX - 4}" y="{d.MB + 9}" text-anchor="end" opacity="0.6">{num(node_gini, 2)}</text>'
        )
        out.append(
            f'<text x="{d.PX - 4}" y="{d.MB + d.MG - 2}" text-anchor="end" opacity="0.6">0</text>'
        )
        for axis in (0, 1):
            curve = " ".join(
                "{:.1f},{:.1f}".format(*at(mark))
                for mark in marks
                if mark[0] == axis
            )
            if curve:
                out.append(
                    f'<polyline points="{curve}" fill="none" stroke="currentColor" stroke-opacity="0.6" stroke-width="1.2" stroke-linejoin="round"/>'
                )
        if best:
            out.append(
                '<circle cx="{:.1f}" cy="{:.1f}" r="5" fill="none" stroke="{}" stroke-width="1.8"/>'.format(
                    *at(best), d.green
                )
            )
        if scanning:
            out.append(
                '<circle cx="{:.1f}" cy="{:.1f}" r="3"/>'.format(
                    *at(marks[-1])
                )
            )

    # --- below the plot: what the Gini measures, for the node then for each side of the threshold
    top = d.MB + d.MG + 34
    out.append(
        f'<text x="218" y="{top - 6}" text-anchor="end" opacity="0.6">points</text>'
    )
    out.append(
        f'<text x="270" y="{top - 6}" text-anchor="end" opacity="0.6">Gini</text>'
    )
    out.append(
        d.composition(top, "node", labels, len(rows), num(node_gini))
    )
    if scanning:
        out.append(
            d.composition(
                top + 18, "yes", left, len(rows), num(gini(left))
            )
        )
        out.append(
            d.composition(
                top + 36, "no", right, len(rows), num(gini(right))
            )
        )
        out.append(
            f'<text x="38" y="{top + 68}" font-size="11">{len(left)}/{len(rows)} × {num(gini(left))} + '
            f'{len(right)}/{len(rows)} × {num(gini(right))} = <tspan font-weight="600">{num(score)}</tspan></text>'
        )

    # --- the tree: the branch of the node just created grows, a leaf just decided blossoms
    newborn = (
        path
        if node is not None and not decided and threshold is None
        else None
    )
    popped = path if node is not None and "leaf" in node else None
    out.append(
        d.tree(
            limits[0],
            current=path if node is not None else None,
            newborn=newborn,
            popped=popped,
        )
    )
    return "".join(out) + "</svg>"


@app.function(hide_code=True)
def view_predict(point, nodes, y, path=None, labels=None, *, X, max_depth):
    """Draws a state of `predict_proba`: the point, the region it is in, its path in the tree."""
    d = Drawing(X, y, nodes)
    here = path or ""
    node = nodes[here]
    group = y[node["rows"]]
    shares = np.bincount(group, minlength=d.n_classes) / len(group)
    where = f"node {here}" if here else "the root"

    # --- three lines of text: where the point is, the question asked, the class shares
    said = [f"Point ({num(point[0])}, {num(point[1])}) at {where}"]
    said[0] += f": {len(group)} points, Gini {num(gini(group))}"
    if "split" in node:
        feature, threshold = node["split"][:2]
        yes = point[feature] <= threshold
        said.append(
            f"Question: {rule(feature, threshold)}? {'Yes' if yes else 'No'}, because x{'₁₂'[feature]} = {num(point[feature])}: "
            f"it goes down to the {'left' if yes else 'right'}"
        )
    else:
        said.append("This is a leaf: no more question to ask")
    listed = ", ".join(
        f"{'ABC'[label]} {num(share, 3)}"
        for label, share in enumerate(shares)
    )
    said.append(
        (
            "predict_proba returns: "
            if labels is not None
            else "Class shares at this node: "
        )
        + listed
    )

    # --- the scatter plot: every region, the current node's one outlined, the point to classify
    out = [
        d.start(said),
        d.frames(margins=False),
        d.leaf_regions(),
        d.points(node["rows"]),
        d.split_lines(here),
    ]
    out.append(
        d.rect(
            d.box(here),
            'fill="none" stroke="currentColor" stroke-opacity="0.8" stroke-width="1.3"',
        )
    )
    cx, cy = d.sx(point[0]), d.sy(point[1])
    out.append(
        f'<path d="M{cx - 9:.1f},{cy:.1f}h18M{cx:.1f},{cy - 9:.1f}v18" stroke="currentColor" stroke-width="1.6"/>'
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="none" stroke="currentColor" stroke-width="2"/>'
    )

    # --- below the plot: the composition of the group at each node crossed, from the root to the current node
    top = d.PY + d.PS + 44
    out.append(
        f'<text x="218" y="{top - 6}" text-anchor="end" opacity="0.6">points</text>'
    )
    out.append(
        f'<text x="270" y="{top - 6}" text-anchor="end" opacity="0.6">leading</text>'
    )
    for depth in range(len(here) + 1):
        crossed = y[nodes[here[:depth]]["rows"]]
        counts = np.bincount(crossed, minlength=d.n_classes)
        leading = f"{'ABC'[counts.argmax()]} {num(counts.max() / len(crossed), 2)}"
        out.append(
            d.composition(
                top + 18 * depth,
                here[:depth] or "root",
                crossed,
                d.n_total,
                leading,
            )
        )

    # --- the tree: only the path followed stays vivid
    out.append(
        d.tree(
            max_depth,
            current=here,
            lit={here[:depth] for depth in range(len(here) + 1)},
        )
    )
    return "".join(out) + "</svg>"


if __name__ == "__main__":
    app.run()
