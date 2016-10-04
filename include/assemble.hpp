#include <string>
#include <fstream>
#include <cassert>
#include "include/global.hpp"
#include "include/csrmatrixcpu.hpp"

struct TriangleQ2 { size_t ID; size_t nodeA; size_t nodeB; size_t nodeC; size_t nodeD; size_t nodeE; size_t nodeF; };

void mesh_q2(string filename, std::vector<Node>& nodes, std::vector<TriangleQ2>& elements, size_t& highest_edgenode);

void assemble_id(CsrMatrixCpu& matrix, pstd::vector<Node>& nodes, std::vector<TriangleQ2>& elements);
