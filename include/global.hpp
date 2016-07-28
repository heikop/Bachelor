#ifndef __GLOBAL_HPP_
#define __GLOBAL_HPP_

struct Node { size_t ID; float x; float y; };
struct Triangle { size_t ID; size_t nodeA; size_t nodeB; size_t nodeC; };
struct FullTriangle { size_t ID; Node nodeA; Node nodeB; Node nodeC; };

#endif
