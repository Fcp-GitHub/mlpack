from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.inspection import DecisionBoundaryDisplay

from mlpack import model


def _compute_grid_prop(
        num_subplots_x: int | None,
        num_subplots_y: int | None,
        feat_pairs
) -> tuple[int, int]:
    """
    Internal utility method that computes the number of rows and columns
    of a grid of subplots, given its inputs.
    """

    # If grid parameters are `None` compute them automatically
    if num_subplots_x is None: 
        num_subplots_x = int( np.ceil( np.sqrt(feat_pairs) ) )

    if num_subplots_y is None:    
        num_subplots_y = int( np.ceil( feat_pairs/num_subplots_x ) )

    return num_subplots_x, num_subplots_y


def decision_boundary_grid(
        estimator: model.Model,
        X: np.ndarray,
        num_subplots_x: int | None = None,
        num_subplots_y: int | None = None
):
    """
    Generate a grid of pairwise subplots showing the decision
    boundary of a classifier and a samples' scatterplot.
    """

    def _one_subplot(ax: plt.Axes, estimator: model.Model, X: np.ndarray):
        """
        Generate a single subplot given `Axes` object `ax`.
        """
        
        DecisionBoundaryDisplay.from_estimator(
            estimator = estimator,
            X = X,
            response_method = "predict",
            ax = ax,
            alpha = 0.5
        )

        return ax.scatter(X[:, 0], X[:, 1])
    
    # Get combinations without repetitions of features' indices
    feat_pairs = combinations(range(X.shape[1]), 2)

    # Compute automatically best values for grid size
    num_subplots_x, num_subplots_y = _compute_grid_prop(
            num_subplots_x, num_subplots_y,
            feat_pairs
    ) 

    # Create `plt.Figure` and `plt.Axes` instances
    fig, axs = plt.subplots(num_subplots_x, num_subplots_y)

    for row, col in feat_pairs:
        _one_subplot(
                axs[row,col],
                estimator,
                X
        )

    return


def animate_decision_boundary(
        estimators: list[model.Model],
        X: np.ndarray,
        num_subplots_x: int | None = None,
        num_subplots_y: int | None = None
) -> FuncAnimation:
    """
    Generate the animated evolution of a linear classifier's decision boundary.


    Parameters
    ----------

    estimators: {array-like} of `Model` instances
        List of partially-trained estimators.
    X: `ndarray` of shape (classes, features)
        Samples array.
    num_subplots_x: `int` or `None`, default=`None`
        Number of desired rows in the final subplots grid.
        If `None`, it is automatically calculated as the maximum
        number of rows based on `X`.
    num_subplots_y: `int` or `None`, default=`None`
        Number of desired columns in the final subplots grid.
        If `None`, it is automatically calculated as the maximum
        number of columns based on `X`.


    Returns
    -------
    `FuncAnimation` instance that **must be stored** in a variable
    in order to have the animation displayed.
    """

    def init_func():
        """
        Initialize all subplots.
        """

        # Array of all `PathCollections` returned by `scatter()`
        scatters = []

        # For all subplots...
        for row, col in feat_pairs:

            # Plot the specific subplot
            scatters.append(_one_subplot(
                axs[row, col],
                estimators[0],
                X[row, col])
            )

        return scatters

    def update(frame):
        
        # Array of all `PathCollections` returned by `scatter()`
        scatters = []

        # For all subplots
        for row, col in feat_pairs:

            # Plot specific subplot with updated predictions
            scatters.append(_one_subplot(
                axs[row, col],
                estimators[frame],
                X[row, col])
            )

        return scatters

    def _one_subplot(ax: plt.Axes, estimator: model.Model, X: np.ndarray):
        """
        Generate a single subplot given `Axes` object `ax`.
        """
        
        DecisionBoundaryDisplay.from_estimator(
            estimator = estimator,
            X = X,
            response_method = "predict",
            ax = ax,
            alpha = 0.5
        )

        return ax.scatter(X[:, 0], X[:, 1])
    
    # Get combinations without repetitions of features' indices
    feat_pairs = combinations(range(X.shape[1]), 2)

    # Compute automatically best values for number of rows and columns of the grid
    num_subplots_x, num_subplots_y = _compute_grid_prop(
            num_subplots_x, num_subplots_y,
            feat_pairs
    )

    # Instantiate a grid of plots
    fig, axs = plt.subplots(nrows = num_subplots_x, ncols = num_subplots_y)

    return FuncAnimation(
            fig = fig,
            func = update,
            frames = len(),
            init_func = init_func,
            blit = True,
            repeat = True,
            interval = 750
    )

def interactive_decision_boundary(
        num_subplots_x: int | None = None,
        num_subplots_y: int | None = None
) -> FuncAnimation:
    """
    Generate an interactive plot about the evolution of a linear classifier's decision boundary.


    Parameters
    ----------
    """
    pass

def predictions_plot(X: np.ndarray, y_pred: np.ndarray, y_exp: np.ndarray, title="", num_images=64, num_subplots_x=8, num_subplots_y=8):
    """
    Plot digits with predictions of color green if y_pred = y_exp or red otherwise.

    
    Parameters
    ----------
    X: samples' matrix.
    y_pred: classifier's predictions.
    y_exp: expected results.
    num_images: number of digits to display.
    num_subplots_x: number of digits to display per row.
    num_subplots_y: number of digits to display per column.


    Notes
    -----
    `num_subplots_x`*`num_subplots_y` must be less or equal to `num_images`
    """

    def _convert_to_np(x):
        if not isinstance(x, np.ndarray):
            return np.asarray(x)
        else:
            return x

    _X      = _convert_to_np(X)
    _y_pred = _convert_to_np(y_pred)
    _y_exp  = _convert_to_np(y_exp)

    # Generate plt.Figure and adjust subplots' layout
    _fig = plt.figure(num=title, figsize=(6, 6))
    _fig.subplots_adjust(left=0, right=1, bottom=0, top=1,
                         hspace=0.05, wspace=0.05
                        )

    # Plot random data
    _indices = np.random.randint(low=0, high=_y_exp.shape[0],
                                 size=num_images
                                )

    for im,i in zip(_indices, range(num_images)):
        _ax = _fig.add_subplot(num_subplots_x, num_subplots_y, i + 1, xticks=[], yticks=[])
        # Reshape _X to an array of unknown elements "along the x axis" (number of samples), 8 elements "along the y axis" and 8 elements "along the z axis" (array of 8x8 pixel matrices)
        _ax.imshow(_X.reshape(-1,8,8)[im], cmap=plt.cm.binary, interpolation='nearest')

        if _y_pred[im] == _y_exp[im]:
            _ax.text(0, 7, str(_y_pred[im]), color='g')
        else:
            _ax.text(0, 7, str(_y_pred[im]), color='r')
