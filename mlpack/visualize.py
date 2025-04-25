import matplotlib.pyplot as plt
import numpy as np

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
