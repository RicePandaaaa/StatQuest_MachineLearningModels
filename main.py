from xgboost.RegressionTree import RegressionTree

import numpy as np

def main():
    """
    Main function
    """

    # Create the data
    data = np.array([1, 2, 3, 4, 5])
    predictions = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    # Create the tree
    tree = RegressionTree(data, predictions, 0.1, 0.1, max_depth=3)

    # Traverse the tree
    nodes = [tree.root]

    while nodes:
        node = nodes.pop(0)
        print(node)
        
        if node.get_left():
            nodes.append(node.get_left())
        if node.get_right():
            nodes.append(node.get_right())


if __name__ == "__main__":
    print(float('inf') == float('inf'))
    main()