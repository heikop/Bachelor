#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <typeinfo>
#include <functional>

#include "../../include/elements.hpp"
#include "../../include/readfile.hpp"
#include "../../include/structure.hpp"
#include "../../include/quadrature.hpp"
#include "../../include/cgsolver.hpp"

#include "../../include/csrmatrixcpu.hpp"
#include "../../include/vectorcpu.hpp"

#include <omp.h>
#include <stdio.h>

using namespace std;

int main(int argc, char* argv[])
{
    const size_t OUTPUTSIZE = 10;
    if (argc > 1)
        std::cout << "incl solving and output" << std::endl;

    float walltime[2];
    std::cout << ">- CPU: MESH FORMAT, Q2 -<" << std::endl;
    std::vector<Vertex<float>> nodes;
    std::vector<Element<float>*> elements;

    file_to_mesh_all("../data/square_quadrilateral_q1_b.msh", nodes, elements);
    std::cout << "num nodes: " << nodes.size() << std::endl;
    std::cout << "num elements: " << elements.size() << std::endl;

    CsrMatrixCpu<float> mat(nodes.size());
    float* localmatrices = new float[16*elements.size()];
    std::cout << "structure" << std::flush;
    walltime[0] = omp_get_wtime();
    structure(mat, elements);
    walltime[0] -= omp_get_wtime();
    std::cout << " - done (" << -walltime[0] * 1000.0 << ")" << std::endl;
    std::cout << "assemble" << std::flush;
    walltime[1] = omp_get_wtime();

    float qp{std::sqrt(0.6)};
    float weight[9] = {25.0/81.0*0.0625, 40.0/81.0*0.0625, 25.0/81.0*0.0625,
                       40.0/81.0*0.0625, 64.0/81.0*0.0625, 40.0/81.0*0.0625,
                       25.0/81.0*0.0625, 40.0/81.0*0.0625, 25.0/81.0*0.0625};
    float quadpoint[9][2] = {{ -qp,  -qp},
                             { -qp, 0.0f},
                             { -qp,   qp},
                             {0.0f,  -qp},
                             {0.0f, 0.0f},
                             {0.0f,   qp},
                             {  qp,  -qp},
                             {  qp, 0.0f},
                             {  qp,   qp}};
    //float grad[9][8] = {{-(1.0f-quadpoint[0][1]), -(1.0f-quadpoint[0][0]),   // qp[0], i=0
    //                      (1.0f-quadpoint[0][1]), -(1.0f+quadpoint[0][0]),   // qp[0], i=1
    //                      (1.0f+quadpoint[0][1]),  (1.0f+quadpoint[0][0]),   // qp[0], i=2
    //                     -(1.0f+quadpoint[0][1]),  (1.0f-quadpoint[0][0])},  // qp[0], i=3
    //                    {-(1.0f-quadpoint[1][1]), -(1.0f-quadpoint[1][0]),   // qp[1], i=0
    //                      (1.0f-quadpoint[1][1]), -(1.0f+quadpoint[1][0]),   // qp[1], i=1
    //                      (1.0f+quadpoint[1][1]),  (1.0f+quadpoint[1][0]),   // qp[1], i=2
    //                     -(1.0f+quadpoint[1][1]),  (1.0f-quadpoint[1][0])},  // qp[1], i=3
    //                    {-(1.0f-quadpoint[2][1]), -(1.0f-quadpoint[2][0]),   // qp[2], i=0
    //                      (1.0f-quadpoint[2][1]), -(1.0f+quadpoint[2][0]),   // qp[2], i=1
    //                      (1.0f+quadpoint[2][1]),  (1.0f+quadpoint[2][0]),   // qp[2], i=2
    //                     -(1.0f+quadpoint[2][1]),  (1.0f-quadpoint[2][0])},  // qp[2], i=3
    //                    {-(1.0f-quadpoint[3][1]), -(1.0f-quadpoint[3][0]),   // qp[3], i=0
    //                      (1.0f-quadpoint[3][1]), -(1.0f+quadpoint[3][0]),   // qp[3], i=1
    //                      (1.0f+quadpoint[3][1]),  (1.0f+quadpoint[3][0]),   // qp[3], i=2
    //                     -(1.0f+quadpoint[3][1]),  (1.0f-quadpoint[3][0])},  // qp[3], i=3
    //                    {-(1.0f-quadpoint[4][1]), -(1.0f-quadpoint[4][0]),   // qp[4], i=0
    //                      (1.0f-quadpoint[4][1]), -(1.0f+quadpoint[4][0]),   // qp[4], i=1
    //                      (1.0f+quadpoint[4][1]),  (1.0f+quadpoint[4][0]),   // qp[4], i=2
    //                     -(1.0f+quadpoint[4][1]),  (1.0f-quadpoint[4][0])},  // qp[4], i=3
    //                    {-(1.0f-quadpoint[5][1]), -(1.0f-quadpoint[5][0]),   // qp[5], i=0
    //                      (1.0f-quadpoint[5][1]), -(1.0f+quadpoint[5][0]),   // qp[5], i=1
    //                      (1.0f+quadpoint[5][1]),  (1.0f+quadpoint[5][0]),   // qp[5], i=2
    //                     -(1.0f+quadpoint[5][1]),  (1.0f-quadpoint[5][0])},  // qp[5], i=3
    //                    {-(1.0f-quadpoint[6][1]), -(1.0f-quadpoint[6][0]),   // qp[6], i=0
    //                      (1.0f-quadpoint[6][1]), -(1.0f+quadpoint[6][0]),   // qp[6], i=1
    //                      (1.0f+quadpoint[6][1]),  (1.0f+quadpoint[6][0]),   // qp[6], i=2
    //                     -(1.0f+quadpoint[6][1]),  (1.0f-quadpoint[6][0])},  // qp[6], i=3
    //                    {-(1.0f-quadpoint[7][1]), -(1.0f-quadpoint[7][0]),   // qp[7], i=0
    //                      (1.0f-quadpoint[7][1]), -(1.0f+quadpoint[7][0]),   // qp[7], i=1
    //                      (1.0f+quadpoint[7][1]),  (1.0f+quadpoint[7][0]),   // qp[7], i=2
    //                     -(1.0f+quadpoint[7][1]),  (1.0f-quadpoint[7][0])},  // qp[7], i=3
    //                    {-(1.0f-quadpoint[8][1]), -(1.0f-quadpoint[8][0]),   // qp[8], i=0
    //                      (1.0f-quadpoint[8][1]), -(1.0f+quadpoint[8][0]),   // qp[8], i=1
    //                      (1.0f+quadpoint[8][1]),  (1.0f+quadpoint[8][0]),   // qp[8], i=2
    //                     -(1.0f+quadpoint[8][1]),  (1.0f-quadpoint[8][0])}}; // qp[8], i=3

    // function begin: assemble
    const size_t num_elem{elements.size()};

    #pragma omp parallel for num_threads(16)
    for (size_t e=0; e < num_elem; e++)
    {
        const std::vector<size_t> vertexids{elements[e]->vertexids()};
        std::array<float, 8> coords = static_cast<QuadrilateralQ1<float>*>(elements[e])->get_pointcoords();
        for (size_t i{0}; i < 4; ++i)
        {
            for (size_t j{0}; j < 4; ++j)
            {
                float val{0.0};

                for (size_t p{0}; p < 9; ++p)
                {
                    float xi  = quadpoint[p][0];
                    float eta = quadpoint[p][1];
                    float B[2][2] =
                        { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) ,
                            ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) },
                          { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) ,
                            ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) } };
                    //    { { ( grad[p][0]*coords[0] + grad[p][2]*coords[1] + grad[p][4]*coords[2] + grad[p][6]*coords[3] ) ,
                    //        ( grad[p][1]*coords[0] + grad[p][3]*coords[1] + grad[p][5]*coords[2] + grad[p][7]*coords[3] ) },
                    //      { ( grad[p][0]*coords[4] + grad[p][2]*coords[5] + grad[p][4]*coords[6] + grad[p][6]*coords[7] ) ,
                    //        ( grad[p][1]*coords[4] + grad[p][3]*coords[5] + grad[p][5]*coords[6] + grad[p][7]*coords[7] ) } };

                    // help vars
                    std::array<float, 2> grad1 = (i == 0 ? std::array<float, 2>{-(1.0f - eta),
                                                                                -(1.0f - xi )} :
                                                 (i == 1 ? std::array<float, 2>{ (1.0f - eta),
                                                                                -(1.0f + xi )} :
                                                 (i == 2 ? std::array<float, 2>{ (1.0f + eta),
                                                                                 (1.0f + xi )} :
                                                           std::array<float, 2>{-(1.0f + eta),
                                                                                 (1.0f - xi )} ) ) );
                    std::array<float, 2> grad2 = (j == 0 ? std::array<float, 2>{-(1.0f - eta),
                                                                                -(1.0f - xi )} :
                                                 (j == 1 ? std::array<float, 2>{ (1.0f - eta),
                                                                                -(1.0f + xi )} :
                                                 (j == 2 ? std::array<float, 2>{ (1.0f + eta),
                                                                                 (1.0f + xi )} :
                                                           std::array<float, 2>{-(1.0f + eta),
                                                                                 (1.0f - xi )} ) ) );

                    val  += weight[p]
                            * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
                                + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
                    //        * (   ( B[1][1] * grad[p][2*i] - B[1][0] * grad[p][2*i+1]) * ( B[1][1] * grad[p][2*j] - B[1][0] * grad[p][2*j+1])
                    //            + (-B[0][1] * grad[p][2*i] + B[0][0] * grad[p][2*i+1]) * (-B[0][1] * grad[p][2*j] + B[0][0] * grad[p][2*j+1]) )
                            / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
                } // end for p (quadrature point)
                //localmatrices[16*e + 4*i + j] = val;
                mat.add(vertexids[i], vertexids[j], val);
            } // end for j
        } // end for i
    } // end for elements


//    #pragma omp parallel for num_threads(16)
//    for (size_t e=0; e < num_elem; e++)
//    {
//        const std::vector<size_t> vertexids{elements[e]->vertexids()};
//        std::array<float, 8> coords = static_cast<QuadrilateralQ1<float>*>(elements[e])->get_pointcoords();
//
//        // i = 0, j = 0
//        float val{0.0};
//        for (size_t p{0}; p < 9; ++p)
//        {
//            float xi  = quadpoint[p][0];
//            float eta = quadpoint[p][1];
//            float B[2][2] =
//            { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) ,
//                  ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) },
//            { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) ,
//                ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) } };
//
//            // help vars
//            std::array<float, 2> grad1 = {-(1.0f - eta),
//                -(1.0f - xi )};
//            std::array<float, 2> grad2 = {-(1.0f - eta),
//                -(1.0f - xi )};
//
//            val  += weight[p]
//                * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
//                        + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
//                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
//        } // end for p (quadrature point)
//        localmatrices[16*e] = val;
//        //mat.add(vertexids[0], vertexids[0], val);
//
//        // i = 0, j = 1
//        val = 0.0f;
//        for (size_t p{0}; p < 9; ++p)
//        {
//            float xi  = quadpoint[p][0];
//            float eta = quadpoint[p][1];
//            float B[2][2] =
//            { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) ,
//                  ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) },
//            { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) ,
//                ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) } };
//
//            // help vars
//            std::array<float, 2> grad1 = {-(1.0f - eta),
//                -(1.0f - xi )};
//            std::array<float, 2> grad2 = { (1.0f - eta),
//                -(1.0f + xi )};
//
//            val  += weight[p]
//                * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
//                        + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
//                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
//        } // end for p (quadrature point)
//        localmatrices[16*e + 1] = val;
//        //mat.add(vertexids[0], vertexids[1], val);
//
//        // i = 0, j = 2
//        val = 0.0f;
//        for (size_t p{0}; p < 9; ++p)
//        {
//            float xi  = quadpoint[p][0];
//            float eta = quadpoint[p][1];
//            float B[2][2] =
//            { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) ,
//                  ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) },
//            { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) ,
//                ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) } };
//
//            // help vars
//            std::array<float, 2> grad1 = {-(1.0f - eta),
//                -(1.0f - xi )};
//            std::array<float, 2> grad2 = { (1.0f + eta),
//                (1.0f + xi )};
//
//            val  += weight[p]
//                * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
//                        + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
//                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
//        } // end for p (quadrature point)
//        localmatrices[16*e + 2] = val;
//        //mat.add(vertexids[0], vertexids[2], val);
//
//        // i = 0, j = 3
//        val = 0.0f;
//        for (size_t p{0}; p < 9; ++p)
//        {
//            float xi  = quadpoint[p][0];
//            float eta = quadpoint[p][1];
//            float B[2][2] =
//            { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) ,
//                  ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) },
//            { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) ,
//                ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) } };
//
//            // help vars
//            std::array<float, 2> grad1 = {-(1.0f - eta),
//                -(1.0f - xi )};
//            std::array<float, 2> grad2 = {-(1.0f + eta),
//                (1.0f - xi )};
//
//            val  += weight[p]
//                * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
//                        + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
//                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
//        } // end for p (quadrature point)
//        localmatrices[16*e + 3] = val;
//        //mat.add(vertexids[0], vertexids[3], val);
//
//        // i = 1, j = 0
//        val = 0.0f;
//        for (size_t p{0}; p < 9; ++p)
//        {
//            float xi  = quadpoint[p][0];
//            float eta = quadpoint[p][1];
//            float B[2][2] =
//            { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) ,
//                  ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) },
//            { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) ,
//                ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) } };
//
//            // help vars
//            std::array<float, 2> grad1 = { (1.0f - eta),
//                -(1.0f + xi )};
//            std::array<float, 2> grad2 = {-(1.0f - eta),
//                -(1.0f - xi )};
//
//            val  += weight[p]
//                * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
//                        + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
//                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
//        } // end for p (quadrature point)
//        localmatrices[16*e + 4] = val;
//        //mat.add(vertexids[1], vertexids[0], val);
//
//        // i = 1, j = 1
//        val = 0.0f;
//        for (size_t p{0}; p < 9; ++p)
//        {
//            float xi  = quadpoint[p][0];
//            float eta = quadpoint[p][1];
//            float B[2][2] =
//            { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) ,
//                  ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) },
//            { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) ,
//                ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) } };
//
//            // help vars
//            std::array<float, 2> grad1 = { (1.0f - eta),
//                -(1.0f + xi )};
//            std::array<float, 2> grad2 = { (1.0f - eta),
//                -(1.0f + xi )};
//
//            val  += weight[p]
//                * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
//                        + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
//                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
//        } // end for p (quadrature point)
//        localmatrices[16*e + 5] = val;
//        //mat.add(vertexids[1], vertexids[1], val);
//
//        // i = 1, j = 2
//        val = 0.0f;
//        for (size_t p{0}; p < 9; ++p)
//        {
//            float xi  = quadpoint[p][0];
//            float eta = quadpoint[p][1];
//            float B[2][2] =
//            { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) ,
//                  ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) },
//            { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) ,
//                ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) } };
//
//            // help vars
//            std::array<float, 2> grad1 = { (1.0f - eta),
//                -(1.0f + xi )};
//            std::array<float, 2> grad2 = { (1.0f + eta),
//                (1.0f + xi )};
//
//            val  += weight[p]
//                * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
//                        + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
//                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
//        } // end for p (quadrature point)
//        localmatrices[16*e + 6] = val;
//        //mat.add(vertexids[1], vertexids[2], val);
//
//        // i = 1, j = 3
//        val = 0.0f;
//        for (size_t p{0}; p < 9; ++p)
//        {
//            float xi  = quadpoint[p][0];
//            float eta = quadpoint[p][1];
//            float B[2][2] =
//            { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) ,
//                  ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) },
//            { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) ,
//                ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) } };
//
//            // help vars
//            std::array<float, 2> grad1 = { (1.0f - eta),
//                -(1.0f + xi )};
//            std::array<float, 2> grad2 = {-(1.0f + eta),
//                (1.0f - xi )};
//
//            val  += weight[p]
//                * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
//                        + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
//                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
//        } // end for p (quadrature point)
//        localmatrices[16*e + 7] = val;
//        //mat.add(vertexids[1], vertexids[3], val);
//
//        // i = 2, j = 0
//        val = 0.0f;
//        for (size_t p{0}; p < 9; ++p)
//        {
//            float xi  = quadpoint[p][0];
//            float eta = quadpoint[p][1];
//            float B[2][2] =
//            { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) ,
//                  ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) },
//            { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) ,
//                ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) } };
//
//            // help vars
//            std::array<float, 2> grad1 = { (1.0f + eta),
//                (1.0f + xi )};
//            std::array<float, 2> grad2 = {-(1.0f - eta),
//                -(1.0f - xi )};
//
//            val  += weight[p]
//                * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
//                        + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
//                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
//        } // end for p (quadrature point)
//        localmatrices[16*e + 8] = val;
//        //mat.add(vertexids[2], vertexids[0], val);
//
//        // i = 2, j = 1
//        val = 0.0f;
//        for (size_t p{0}; p < 9; ++p)
//        {
//            float xi  = quadpoint[p][0];
//            float eta = quadpoint[p][1];
//            float B[2][2] =
//            { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) ,
//                  ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) },
//            { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) ,
//                ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) } };
//
//            // help vars
//            std::array<float, 2> grad1 = { (1.0f + eta),
//                (1.0f + xi )};
//            std::array<float, 2> grad2 = { (1.0f - eta),
//                -(1.0f + xi )};
//
//            val  += weight[p]
//                * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
//                        + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
//                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
//        } // end for p (quadrature point)
//        localmatrices[16*e + 9] = val;
//        //mat.add(vertexids[2], vertexids[1], val);
//
//        // i = 2, j = 2
//        val = 0.0f;
//        for (size_t p{0}; p < 9; ++p)
//        {
//            float xi  = quadpoint[p][0];
//            float eta = quadpoint[p][1];
//            float B[2][2] =
//            { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) ,
//                  ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) },
//            { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) ,
//                ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) } };
//
//            // help vars
//            std::array<float, 2> grad1 = { (1.0f + eta),
//                (1.0f + xi )};
//            std::array<float, 2> grad2 = { (1.0f + eta),
//                (1.0f + xi )};
//
//            val  += weight[p]
//                * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
//                        + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
//                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
//        } // end for p (quadrature point)
//        localmatrices[16*e + 10] = val;
//        //mat.add(vertexids[2], vertexids[2], val);
//
//        // i = 2, j = 3
//        val = 0.0f;
//        for (size_t p{0}; p < 9; ++p)
//        {
//            float xi  = quadpoint[p][0];
//            float eta = quadpoint[p][1];
//            float B[2][2] =
//            { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) ,
//                  ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) },
//            { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) ,
//                ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) } };
//
//            // help vars
//            std::array<float, 2> grad1 = { (1.0f + eta),
//                (1.0f + xi )};
//            std::array<float, 2> grad2 = {-(1.0f + eta),
//                (1.0f - xi )};
//
//            val  += weight[p]
//                * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
//                        + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
//                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
//        } // end for p (quadrature point)
//        localmatrices[16*e + 11] = val;
//        //mat.add(vertexids[2], vertexids[3], val);
//
//        // i = 3, j = 0
//        val = 0.0f;
//        for (size_t p{0}; p < 9; ++p)
//        {
//            float xi  = quadpoint[p][0];
//            float eta = quadpoint[p][1];
//            float B[2][2] =
//            { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) ,
//                  ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) },
//            { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) ,
//                ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) } };
//
//            // help vars
//            std::array<float, 2> grad1 = {-(1.0f + eta),
//                (1.0f - xi )};
//            std::array<float, 2> grad2 = {-(1.0f - eta),
//                -(1.0f - xi )};
//
//            val  += weight[p]
//                * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
//                        + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
//                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
//        } // end for p (quadrature point)
//        localmatrices[16*e + 12] = val;
//        //mat.add(vertexids[3], vertexids[0], val);
//
//        // i = 3, j = 1
//        val = 0.0f;
//        for (size_t p{0}; p < 9; ++p)
//        {
//            float xi  = quadpoint[p][0];
//            float eta = quadpoint[p][1];
//            float B[2][2] =
//            { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) ,
//                  ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) },
//            { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) ,
//                ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) } };
//
//            // help vars
//            std::array<float, 2> grad1 = {-(1.0f + eta),
//                (1.0f - xi )};
//            std::array<float, 2> grad2 = { (1.0f - eta),
//                -(1.0f + xi )};
//
//            val  += weight[p]
//                * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
//                        + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
//                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
//        } // end for p (quadrature point)
//        localmatrices[16*e + 13] = val;
//        //mat.add(vertexids[3], vertexids[1], val);
//
//        // i = 3, j = 2
//        val = 0.0f;
//        for (size_t p{0}; p < 9; ++p)
//        {
//            float xi  = quadpoint[p][0];
//            float eta = quadpoint[p][1];
//            float B[2][2] =
//            { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) ,
//                  ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) },
//            { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) ,
//                ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) } };
//
//            // help vars
//            std::array<float, 2> grad1 = {-(1.0f + eta),
//                (1.0f - xi )};
//            std::array<float, 2> grad2 = { (1.0f + eta),
//                (1.0f + xi )};
//
//            val  += weight[p]
//                * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
//                        + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
//                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
//        } // end for p (quadrature point)
//        localmatrices[16*e + 14] = val;
//        //mat.add(vertexids[3], vertexids[2], val);
//
//        // i = 3, j = 3
//        val = 0.0f;
//        for (size_t p{0}; p < 9; ++p)
//        {
//            float xi  = quadpoint[p][0];
//            float eta = quadpoint[p][1];
//            float B[2][2] =
//            { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) ,
//                  ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) },
//            { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) ,
//                ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) } };
//
//            // help vars
//            std::array<float, 2> grad1 = {-(1.0f + eta),
//                (1.0f - xi )};
//            std::array<float, 2> grad2 = {-(1.0f + eta),
//                (1.0f - xi )};
//
//            val  += weight[p]
//                * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
//                        + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
//                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
//        } // end for p (quadrature point)
//        localmatrices[16*e + 15] = val;
//        //mat.add(vertexids[3], vertexids[3], val);
//
//    } // end for elements

    // function end: assemble
    walltime[1] -= omp_get_wtime();
    std::cout << " - done (" << -walltime[1] * 1000.0 << ")" << std::endl;
    std::cout << "num values: " << mat._rowptr[mat._numrows] << std::endl;
    for (int k(0); k < OUTPUTSIZE; ++k)
        std::cout << "(0, " << mat._colind[mat._rowptr[mat._numrows] - k] << ") = " << mat._values[mat._rowptr[mat._numrows] - k] << std::endl;
    std::cout << "extra: " << mat._colind[mat._rowptr[mat._numrows] - 50] << ") = " << mat._values[mat._rowptr[mat._numrows] - 50] << std::endl;
    delete localmatrices;

if (argc > 1)
{
    // assemble rhs
    std::function<float(float, float)> f = [](float x, float y)
                    { return static_cast<float>(2.0) * (x - x*x + y - y*y); };
    size_t numvertices{nodes.size()};
    VectorCpu rhs(numvertices, 0.0);
    for (const auto& e : elements)
    {
        const std::vector<size_t> nodeids = e->vertexids();
        Quadrature<Element, float> quad(e);
        for (size_t i{0}; i < nodeids.size(); ++i)
//            rhs.add(nodeids[i], f(nodes[nodeids[i]].x, nodes[nodeids[i]].y) * quad.integrate_basisfunction(2, i));
            rhs.add(nodeids[i], f(nodes[nodeids[i]].x, nodes[nodeids[i]].y) * quad.integrate_basisfunction(3, i));
    }
    // dirichlet boundary
    for (const auto& n : nodes)
    {
        if (n.x == 0.0 || n.y == 0.0 || n.x == 1.0 || n.y == 1.0)
        {
            for (size_t i{mat._rowptr[n.id]}; i < mat._rowptr[n.id + 1]; ++i)
                mat._values[i] = 0.0f;
            mat.set(n.id, n.id, 1.0f);
            rhs.set(n.id, 0.0f);
        }
    }

    // solve LGS
    std::cout << "solve" << std::flush;
    CgSolver<CsrMatrixCpu<float>, VectorCpu> solver(mat, rhs);
    VectorCpu res(numvertices, 0.1);
    solver.solve(res);
    std::cout << " - done" << std::endl;

    // write vtk-file
    ofstream output("../data/cpu.vtk");
    output << "# vtk DataFile Version 3.0" << std::endl;
    output << "square q2" << std::endl;
    output << "ASCII" << std::endl;
    output << "DATASET UNSTRUCTURED_GRID" << std::endl;
    output << std::endl;
    output << "POINTS " << numvertices << " float" << std::endl;
    for (const auto& n : nodes)
        output << n.x << " " << n.y << " 0" << std::endl;
    output << std::endl;
    output << "CELLS " << elements.size() << " " << 5*elements.size() << std::endl;
    for (const auto& e : elements)
        output << "4 " << e->vertexids()[0] << " " << e->vertexids()[1] << " " << e->vertexids()[2] << " " << e->vertexids()[3] << std::endl;
    output << std::endl;
    output << "CELL_TYPES " << elements.size() << std::endl;
    for (size_t i{0}; i < elements.size(); ++i)
        output << "9" << std::endl; // QuadrilateralQ1
    output << std::endl;
    output << "POINT_DATA " << numvertices << std::endl;
    output << "SCALARS u " << (typeid(float) == typeid(float) ? "float" : "float") << std::endl;
    output << "LOOKUP_TABLE default" << std::endl;
    for (size_t i{0}; i < numvertices; ++i)
        output << (std::abs(res._values[i]) < 0.0001 ? 0 : res._values[i]) << std::endl;
    output.close();
}

    return 0;
}
