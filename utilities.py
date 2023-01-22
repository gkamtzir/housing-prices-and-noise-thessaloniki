import argparse


def read_experiment_parameters():
    """
    Reads the experiment parameters from command line.
    :return: The experiment parameters.
    """
    parser = argparse.ArgumentParser()

    # Setting arguments.
    parser.add_argument("--experiment-id", help="The id of the experiment", default=None)
    parser.add_argument("--experiment-name", help="The name of the experiment", default=None)
    parser.add_argument("--experiment-tags", help="The tags of the experiment", type=str, default=None)
    parser.add_argument("--noise", help="Indicates the noise usage (0 -> No noise, 1 -> Day/Night, 2 -> Day/Night "
                                        "average, 3 -> Day, 4 -> Night", default=0)
    parser.add_argument("--mode", help="The hyperparameter optimization method", default="grid_search")
    parser.add_argument("--area", help="The area of interest", default=None)

    args = parser.parse_args()
    return args.experiment_id, args.experiment_name, args.experiment_tags, \
           int(args.noise), args.mode, args.area
