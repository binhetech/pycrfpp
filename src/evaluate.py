import argparse
from utils import calc_metrics, run_eval

if __name__ == "__main__":
    # Define and parse program input
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     usage="%(prog)s [-h] [options] -file FILE -head HEAD")
    parser.add_argument("-file", help="The path to the labeled file.", required=True)
    parser.add_argument("-head", help="Num of the former character to be ignored.", default=2)
    parser.add_argument("-verbose", help="Whether to print classification report.", default=True)

    args = parser.parse_args()
    # Run the program.
    run_eval(args.file, args.head, args.verbose)
