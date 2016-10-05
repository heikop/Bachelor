#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include "global.hpp"
#include "csrmatrixcpu.hpp"

#include <iostream>

struct TriangleQ2 { size_t ID; size_t nodeA; size_t nodeB; size_t nodeC; size_t nodeD; size_t nodeE; size_t nodeF; };

void mesh_q2(std::string filename, std::vector<Node>& nodes, std::vector<TriangleQ2>& elements, size_t& highest_edgenode);

void structure_id(CsrMatrixCpu& matrix, std::vector<TriangleQ2>& elements);
void assemble_id(CsrMatrixCpu& matrix, std::vector<Node>& nodes, std::vector<TriangleQ2>& elements);
