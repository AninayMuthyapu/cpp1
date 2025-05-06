#include <iostream>
#include "AnyOption/anyoption.h"
int main(int argc, char* argv[]) {
    AnyOption opt;

    // Add options
    opt.addUsage("Usage: ");
    opt.addUsage(" -m <rows of A>");
    opt.addUsage(" -n <cols of B>");
    opt.addUsage(" -k <cols of A / rows of B>");
    opt.setFlag("help", 'h');
    opt.setOption("m", 'm');
    opt.setOption("n", 'n');
    opt.setOption("k", 'k');

    opt.processCommandArgs(argc, argv);

    if (opt.getFlag("help") || !opt.getValue("m") || !opt.getValue("n") || !opt.getValue("k")) {
        opt.printUsage();
        return 1;
    }

    int M = std::stoi(opt.getValue("m"));
    int N = std::stoi(opt.getValue("n"));
    int K = std::stoi(opt.getValue("k"));

    std::cout << "Matrix dimensions:\n";
    std::cout << "A: " << M << "x" << K << "\n";
    std::cout << "B: " << K << "x" << N << "\n";
    std::cout << "C: " << M << "x" << N << "\n";

    return 0;
}
