from utils.libs import *

def get_option_parser():
    op = OptionParser(version="%prog 1.0")
    op.add_option("--report",
              action="store_true", dest="print_report", default=True,
              help="Print a detailed classification report.")
    op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
    op.add_option("--confusion_matrix", default=True,
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
    op.add_option("--permutation_test",
              action="store_true", dest="print_perm_test", default=True,
              help="Print the permutation test's p-value.")
    op.add_option("--scaler",
              action="store", dest="scaler_strategy", default="standardize",
              choices = ["standardize", "normalize"],
              help="Preprocessing Strategy for continuous features [default: %default].")
    op.add_option("--roc_curve",
              action="store_true", dest="print_roc_curve", default=True,
              help="Print roc curve.")
    op.add_option("--precision_recall_curve",
              action="store_true", dest="print_precision_recall_curve", default=True,
              help="Print roc curve.")
    
    return op
