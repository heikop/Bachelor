#include "include/assemblegpu.hpp"

void assemble_gpu_atomic(CsrMatrix& matrix, std::vector<FullTriangle>& elements)
{
    //TODO
    for (auto& elem : elements)
    {
        float B[2][2];
        B[0][0] = elem.nodeB.x - elem.nodeA.x;
        B[1][0] = elem.nodeB.y - elem.nodeA.y;
        B[0][1] = elem.nodeC.x - elem.nodeA.x;
        B[1][1] = elem.nodeC.y - elem.nodeA.y;
        //float d[2];
        //d[0] = elem.nodeA.x;
        //d[1] = elem.nodeA.y;
        float detB(std::abs(B[0][0]*B[1][1] - B[0][1]*B[1][0]));
        float gradA[2], gradB[2], gradC[2]; // in fact these are the gradient multplied by det(B)
        gradA[0] = B[1][0] - B[1][1];
        gradA[1] = B[0][1] - B[0][0];
        gradB[0] = B[1][1];
        gradB[1] = -B[0][1];
        gradC[0] = -B[1][0];
        gradC[1] = B[0][0];

        matrix.add_local(elem.nodeA.ID, elem.nodeA.ID, (gradA[0]*gradA[0] + gradA[1]*gradA[1]) / 2.0 / detB);
        matrix.add_local(elem.nodeA.ID, elem.nodeB.ID, (gradA[0]*gradB[0] + gradA[1]*gradB[1]) / 2.0 / detB);
        matrix.add_local(elem.nodeA.ID, elem.nodeC.ID, (gradA[0]*gradC[0] + gradA[1]*gradC[1]) / 2.0 / detB);
        matrix.add_local(elem.nodeB.ID, elem.nodeA.ID, (gradB[0]*gradA[0] + gradB[1]*gradA[1]) / 2.0 / detB);
        matrix.add_local(elem.nodeB.ID, elem.nodeB.ID, (gradB[0]*gradB[0] + gradB[1]*gradB[1]) / 2.0 / detB);
        matrix.add_local(elem.nodeB.ID, elem.nodeC.ID, (gradB[0]*gradC[0] + gradB[1]*gradC[1]) / 2.0 / detB);
        matrix.add_local(elem.nodeC.ID, elem.nodeA.ID, (gradC[0]*gradA[0] + gradC[1]*gradA[1]) / 2.0 / detB);
        matrix.add_local(elem.nodeC.ID, elem.nodeB.ID, (gradC[0]*gradB[0] + gradC[1]*gradB[1]) / 2.0 / detB);
        matrix.add_local(elem.nodeC.ID, elem.nodeC.ID, (gradC[0]*gradC[0] + gradC[1]*gradC[1]) / 2.0 / detB);
    }
}
