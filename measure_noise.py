import argparse
import numpy as np
from matplotlib import pyplot as plt
import utils
import os


class MeasurePrecision:
    def __init__(self, instrument, constants, design_path, output_path):
        self.constants = constants
        self.instrument = instrument
        self.design_path = design_path
        self.output_path = output_path
        self.design_name = os.path.basename(self.design_path)

    @staticmethod
    def __read_vals(file_name):
        with open(file_name, 'r') as f:
            return list(map(float, f.read().split()))

    def __get_precision_vals(self, selection):
        f = utils.ObjFunctionAverageSquare(self.instrument, self.constants, 1000)
        N = 5
        vals = np.zeros(N)
        for i in range(N):
            f(selection)
            vals[i] = f.sron_precision
        with open(f'{self.output_path}/vals_{self.design_name}', 'w') as file:
            print(*vals, file=file, flush=True)
        return vals

    def __build_boxplot(self, vals=None, file_name=None):
        if file_name:
            vals = MeasurePrecision.__read_vals(file_name)
        fig = plt.figure()
        plt.boxplot(x=vals, vert=False)
        plt.grid()
        fig.savefig(f'{self.output_path}/vals_{self.design_name}_boxplot.pdf')
        plt.close()

    def process(self):
        selection = utils.read_selection(self.design_path)
        vals = self.__get_precision_vals(selection)
        # self.__build_boxplot(vals=vals)


def main():
    parser = argparse.ArgumentParser('Measure of noise in the performance of design')
    parser.add_argument('-d', '--design_path', nargs=1, required=True, type=str,
                        help='Path to the file with sequence of filter ids')
    parser.add_argument('-o', '--out', nargs=1, required=False, default=['.'], type=str,
                        help='Path to the output folder')
    args = parser.parse_args()
    instrument = utils.create_instrument()
    constants = utils.SRONConstants(nCH4=1500, albedo=0.15, sza=70)
    MeasurePrecision(instrument, constants, args.design_path[0], args.out[0]).process()


if __name__ == '__main__':
    main()
