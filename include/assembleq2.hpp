#ifndef __ASSEMBLEQ2_HPP_
#define __ASSEMBLEQ2_HPP_

#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include "global.hpp"
#include "csrmatrixcpu.hpp"

#include <iostream>

struct TriangleQ2 { size_t ID; size_t nodeA; size_t nodeB; size_t nodeC; size_t nodeD; size_t nodeE; size_t nodeF; };
struct FullTriangleQ2 { size_t ID; Node nodeA; Node nodeB; Node nodeC; size_t nodeD; size_t nodeE; size_t nodeF; };
struct NagFlags { size_t num_adjacents; size_t num_midnodes; size_t num_gaps; };

void mesh_q2_full(std::string filename, std::vector<FullTriangleQ2>& elements, size_t& numnodes);
void structure_full(CsrMatrixCpu<double>& matrix, std::vector<FullTriangleQ2>& elements);
void assemble_full(CsrMatrixCpu<double>& matrix, std::vector<FullTriangleQ2>& elements);

void mesh_q2_id(std::string filename, std::vector<Node>& nodes, std::vector<TriangleQ2>& elements, size_t& highest_edgenode);
void structure_id(CsrMatrixCpu<double>& matrix, std::vector<TriangleQ2>& elements);
void assemble_id(CsrMatrixCpu<double>& matrix, std::vector<Node>& nodes, std::vector<TriangleQ2>& elements);

#endif
