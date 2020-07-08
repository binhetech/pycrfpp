from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import argparse


def run_eval(fin, head=2, verbose=True):
    with open(fin, "r", encoding="utf-8") as fo:
        data = fo.readlines()
        y_true, y_pred = [], []
        for i in data:
            i = i.split()
            if i:
                y1 = i[-2][int(head):] if len(i[-2]) > 2 else i[-2]
                y2 = i[-1][int(head):] if len(i[-1]) > 2 else i[-1]
                y_true.append(y1)
                y_pred.append(y2)
        f1 = f1_score(y_true, y_pred, pos_label="LABEL")
        if verbose:
            rp = classification_report(y_true, y_pred)
            print(rp)
        print("f1={}".format(f1))
        return f1


if __name__ == "__main__":
    # Define and parse program input
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     usage="%(prog)s [-h] [options] -file FILE -head HEAD")
    parser.add_argument("-file", help="The path to the labeled file.", required=True)
    parser.add_argument("-head", help="Num of the former character to be ignored.", default=2)

    args = parser.parse_args()
    # Run the program.
    run_eval(args.file, args.head, True)
