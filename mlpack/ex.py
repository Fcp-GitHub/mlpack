import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.contour # Import to check instance type

# 1. Prepare your data
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1, random_state=42)

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("k-NN Decision Boundary Animation")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")

# Plot the data points (these will remain static)
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, s=30, edgecolors='k')

# Initialize the decision boundary plot (will be updated in each frame)
# We don't need to store disp.contour_ (or ._surface, or .surface_)
# as we will identify and remove it from ax.collections directly.
initial_knn = KNeighborsClassifier(n_neighbors=1)
initial_knn.fit(X, y)
DecisionBoundaryDisplay.from_estimator(
    initial_knn,
    X,
    cmap=plt.cm.RdBu,
    alpha=0.6,
    ax=ax,
    plot_method="contourf",
    response_method="predict",
    #shading="auto"
)
# No need for a global contour_collection variable now for removing it.
# The `DecisionBoundaryDisplay.from_estimator` already added it to `ax.collections`.

# Define the range of n_neighbors to animate
n_neighbors_values = range(1, 21, 1) # From 1 to 20 neighbors

# Store the classifiers for each frame (pre-train them to save time during animation)
classifiers = []
for n in n_neighbors_values:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X, y)
    classifiers.append(knn)

# Create a text object to display the current n_neighbors
n_neighbors_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                           verticalalignment='top')

# 4. Create an update function
def update(frame):
    current_n_neighbors = n_neighbors_values[frame]
    current_classifier = classifiers[frame]

    # Iterate through existing collections and remove only the contour plots.
    # It's important to iterate over a copy of the list (ax.collections[:])
    # because you're modifying it during iteration.
    for collection in ax.collections[:]:
        if isinstance(collection, matplotlib.contour.QuadContourSet):
            collection.remove()

    # Create a new DecisionBoundaryDisplay for the current classifier.
    # This automatically adds the new contour plot to `ax.collections`.
    new_disp = DecisionBoundaryDisplay.from_estimator(
        current_classifier,
        X,
        cmap=plt.cm.RdBu,
        alpha=0.6,
        ax=ax,
        plot_method="contourf",
        response_method="predict",
        #shading="auto"
    )

    # Update the n_neighbors text
    n_neighbors_text.set_text(f'n_neighbors = {current_n_neighbors}')

    # Return the artists that were modified.
    # This is the new contour plot and the updated text.
    return [new_disp.surface_, n_neighbors_text] # Use contour_ here for returning, but not for removing.


# 5. Use FuncAnimation
ani = FuncAnimation(fig, update, frames=len(n_neighbors_values),
                    blit=False, repeat=True, interval=200) # interval in ms

# 6. Save or display
plt.show()

# To save as a GIF (requires imagemagick or pillow):
# ani.save('knn_decision_boundary_animation.gif', writer='pillow', fps=5)

# To save as MP4 (requires ffmpeg):
# ani.save('knn_decision_boundary_animation.mp4', writer='ffmpeg', fps=5)
