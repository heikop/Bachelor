#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <typeinfo>
#include <functional>

#include "include/elements.hpp"
#include "include/readfile.hpp"
#include "include/structure.hpp"
#include "include/quadrature.hpp"
#include "include/cgsolver.hpp"

#include "include/csrmatrixcpu.hpp"
#include "include/vectorcpu.hpp"
#include "include/csrmatrixgpu.hpp"
#include "include/vectorgpu.hpp"

#include <omp.h>
#include <stdio.h>

using namespace std;

int main(int argc, char* argv[])
{
    double walltime[2];
    std::cout << ">- CPU: MESH FORMAT, Q2 -<" << std::endl;
    std::vector<Vertex<double>> nodes;
    std::vector<Element<double>*> elements;

    file_to_mesh_all("../data/square_quadrilateral_q1_a.msh", nodes, elements);
    std::cout << "num nodes: " << nodes.size() << std::endl;
    std::cout << "num elements: " << elements.size() << std::endl;

    CsrMatrixCpu<double> mat_cpu(nodes.size());
    std::cout << "structure" << std::flush;
    walltime[0] = omp_get_wtime();
    structure(mat_cpu, elements);
    walltime[0] -= omp_get_wtime();
    std::cout << " - done (" << -walltime[0] * 1000.0 << ")" << std::endl;
    std::cout << "assemble" << std::flush;
    walltime[1] = omp_get_wtime();

    // copy matrix to gpu
    CsrMatrixCpu<double> mat_gpu(nodes.size());
    malloc_cuda(&mat_gpu._colind, mat_cpu._rowptr[mat_cpu._numrows] * sizeof(size_t));
    malloc_cuda(&mat_gpu._values, mat_cpu._rowptr[mat_cpu._numrows] * sizeof(double));
    memcpy_cuda(&mat_gpu._rowptr, &mat_cpu._rowptr, (mat_cpu._numrows + 1) * sizeof(size_t), h2d);
    memcpy_cuda(&mat_gpu._colind, &mat_cpu._colind, mat_cpu._rowptr[mat_cpu._numrows] * sizeof(size_t), h2d);
    memcpy_cuda(&mat_gpu._values, &mat_cpu._values, mat_cpu._rowptr[mat_cpu._numrows] * sizeof(double), h2d);

    // function begin: assemble
    assemble_matrix_Q1_O5(elements, mat_gpu);

    // function end: assemble
    walltime[1] -= omp_get_wtime();
    std::cout << " - done (" << -walltime[1] * 1000.0 << ")" << std::endl;

    return 0;
}
