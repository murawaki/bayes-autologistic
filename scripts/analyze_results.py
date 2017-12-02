import sys
from analyze.analyze_autologistic_mvi_results import analyze_mvi_results
from analyze.analyze_autologistic_param_results import analyze_param_results
from analyze.analyze_autologistic_vh import analyze_vh

if __name__ == '__main__':
    if sys.argv[1] == 'mvi':
        analyze_mvi_results(sys.argv[2])
    elif sys.argv[1] == 'vh':
        analyze_vh(sys.argv[2])
    else:
        analyze_param_results(sys.argv[2])

