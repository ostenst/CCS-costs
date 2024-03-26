def find_points_on_curve(curve, x_values):
    """
    Find the y-values on the curve corresponding to specific x-values.

    Parameters:
        curve (list): List of points representing the curve in the form [[x1, y1], [x2, y2], ...].
        x_values (list): List of x-values for which to find the corresponding y-values.

    Returns:
        dict: A dictionary containing the x-values as keys and their corresponding y-values as values.
    """
    points = {}
    for x_target in x_values:
        # Find the two closest points on the curve to the specified x-value
        closest_points = sorted(curve, key=lambda point: abs(point[0] - x_target))[:2]
        # Perform linear interpolation to find the y-value corresponding to the x-value
        x1, y1 = closest_points[0]
        x2, y2 = closest_points[1]
        y_value = y1 + (y2 - y1) * (x_target - x1) / (x2 - x1)
        # Add the interpolated y-value to the dictionary
        points[x_target] = y_value
    return points

# Define your curve as a list of points [(x, y)]
curve = [
    [10, 25],
    [9, 22],
    [8, 18],
    [7, 15],
    [6, 12],
    [5, 10],
    [4, 8],
    [3, 6],
    [2, 4],
    [1, 2]
]

# Define the x-values for which to find the corresponding y-values
x_low = 3.2
x_high = 8.8
x_values = [x_low, x_high]

# Find the y-values corresponding to the specified x-values
points = find_points_on_curve(curve, x_values)

# Print the result
for x_value, y_value in points.items():
    print(f"For x = {x_value}, y = {y_value:.2f}")
