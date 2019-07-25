import numpy as np
import itertools
np.random.seed(42)


def preprocess(data, y):
    """
    Perform mean normalization on the features and divide the true labels by
    the range of the column. 

    Input:
    - data: Inputs  (n features over m instances).
    - y: True labels.

    Returns a two vales:
    - data: The mean normalized inputs.
    - y: The scaled labels.
    """
    mean = np.mean(data, axis=0)
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    data = np.divide((data - mean), (max_values - min_values))
    y = np.divide((y - np.mean(y)), (np.amax(y) - np.amin(y)))

    return data, y


def compute_cost(data, y, theta):
    """
    Computes the average squared difference between an observation’s actual and
    predicted values for linear regression.  

    Input:
    - data: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.

    Returns a single value:
    - j: the cost associated with the current set of parameters (single number).
    """
    copy_data = np.transpose(data)
    copy_theta = np.diag(theta)
    x = np.transpose(np.array(np.dot(copy_theta, copy_data)))  # theta * X vector
    j = np.sum(x, axis=1)
    j = np.subtract(j, y)  # h - y vector
    j = np.power(j, 2)  # (h - y)^2 vector
    j = np.sum(j)  # sigma on the last vector
    factor = np.divide(1, np.multiply(2, np.size(y)))
    j = np.multiply(factor, j)  # the cost

    return j


def gradient_descent(data, y, theta, alpha, num_iterations):
    """
    Learn the parameters of the model using gradient descent. Gradient descent
    is an optimization algorithm used to minimize some (loss) function by 
    iteratively moving in the direction of steepest descent as defined by the
    negative of the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - data: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iterations: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - j_history: the loss value for every iteration.
    """
    j_history = []  # Use a python list to save cost in every iteration
    copy_data = np.transpose(data)
    theta = np.diag(theta)
    factor = alpha * (1 / y.size)
    for i in range(num_iterations):
        x = np.transpose(np.array(np.dot(theta, copy_data)))
        h = np.sum(x, axis=1)
        parenthesis = np.subtract(h, y)
        for j in range(theta.shape[0]):
            sigma = np.sum(np.multiply(parenthesis, copy_data[j, :]))
            current_theta = np.subtract(theta[j][j], np.multiply(factor, sigma))
            theta[j][j] = current_theta
        j_history.append(compute_cost(data, y, np.diagonal(theta)))

    return np.diagonal(theta), j_history


def pinv(data, y):
    """
    Calculate the optimal values of the parameters using the pseudo-inverse
    approach as you saw in class.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).

    Returns two values:
    - theta: The optimal parameters of your model.

    ########## DO NOT USE numpy.pinv ##############
    """
    x_transpose = np.transpose(data)
    pinv_data = np.dot(x_transpose, data)
    pinv_data = np.linalg.inv(pinv_data)
    pinv_data = np.dot(pinv_data, x_transpose)
    pinv_theta = np.dot(pinv_data, y)

    return pinv_theta


def efficient_gradient_descent(data, y, theta, alpha, num_iterations):
    """
    Learn the parameters of your model, but stop the learning process once
    the improvement of the loss value is smaller than 1e-8. This function is
    very similar to the gradient descent function you already implemented.

    Input:
    - data: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iterations: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    j_history = []  # Use a python list to save cost in every iteration
    copy_data = np.transpose(data)
    improvement = float("inf")
    factor = np.multiply(alpha, np.divide(1, len(y)))
    count = 0
    temp_theta = np.diag(theta)
    while improvement > 1e-8 and count < num_iterations:
        count += 1
        for j in range(theta.size):
            x = np.dot(temp_theta, copy_data)
            h = np.sum(x, axis=0)
            sigma = np.subtract(h, y)
            sigma = np.multiply(sigma, copy_data[j])
            temp_theta[j][j] = temp_theta[j][j] - (factor * np.sum(sigma))
        theta = np.diagonal(temp_theta)
        j_history.append(compute_cost(data, y, theta))
        if count > 1:
            improvement = j_history[-2:][0] - j_history[-1:][0]
        else:
            improvement = j_history[-1:][0]
        if count > 100 and j_history[-100:][0] < j_history[-1:][0]:
            improvement = 0

    return theta, j_history


def find_best_alpha(data, y, iterations):
    """
    Iterate over provided values of alpha and maintain a python dictionary 
    with alpha as the key and the final loss as the value.

    Input:
    - data: a data frame that contains all relevant features.

    Returns:
    - alpha_dict: A python dictionary that hold the loss value after training 
    for every value of alpha.
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}

    for i in range(len(alphas)):
        np.random.seed(42)
        theta = np.random.random(size=2)
        _, j = efficient_gradient_descent(data, y, theta, alphas[i], iterations)
        alpha_dict[alphas[i]] = j[-1:][0]

    return alpha_dict


def generate_triplets(data):
    """
    generate all possible sets of three features out of all relevant features
    available from the given data set X. You might want to use the itertools
    python library.

    Input:
    - data: a data frame that contains all relevant features.

    Returns:
    - A python list containing all feature triplets.
    """

    triplets = list(itertools.combinations(data, 3))

    return triplets


def find_best_triplet(df, triplets, alpha, num_iter):
    """
    Iterate over all possible triplets and find the triplet that best 
    minimizes the cost function. For better performance, you should use the 
    efficient implementation of gradient descent. You should first pre-process
    the data and obtain a array containing the columns corresponding to the
    triplet. Don't forget the bias trick.

    Input:
    - df: A data frame that contains the data
    - triplets: a list of three strings representing three features in X.
    - alpha: The value of the best alpha previously found.
    - num_iterations: The number of updates performed.

    Returns:
    - The best triplet as a python list holding the best triplet as strings.
    """
    best_triplet = None
    best_cost = float("inf")
    y = df["price"].values
    columns_to_drop = ['price', 'id', 'date']
    df = df.drop(columns=columns_to_drop)
    np.random.seed(42)
    theta = np.random.random(size=4)

    for i in range(len(triplets)):
        x = df[np.array(triplets[i])].values
        ones = np.ones_like(np.transpose(x))
        x = np.transpose(np.vstack([ones[0], np.transpose(x)]))
        _, j_history = efficient_gradient_descent(x, y, theta, alpha, num_iter)
        if j_history[-1] < best_cost:
            best_cost = j_history[-1]
            best_triplet = triplets[i]

    return best_triplet
