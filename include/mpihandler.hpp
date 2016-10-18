/*
#ifndef __MPIHANDLER_HPP_
#define __MPIHANDLER_HPP_

#include <mpi.h>
#include <exception>
#include <string>
#include <iostream>
#include <cassert> // only to avoid delete sentences

class MpiHandler
{
public:
    MpiHandler();
    MpiHandler(const MpiHandler&) { assert(false); };// = delete;
//    MpiHandler(MpiHandler&&) = delete;
    ~MpiHandler();
    MpiHandler& operator=(const MpiHandler&) { assert(false); return *this; };// = delete;
//    MpiHandler& operator=(MpiHandler&&) = delete;

    void catch_call(int line, std::string file, const std::exception& ex);

    size_t get_global_size();
    size_t get_global_rank();
    bool is_last();
private:
    size_t _size;
    size_t _rank;
};//class MpiHander

extern MpiHandler __mpi_instance__;

#define MPICALL(X) try { X } \
                   catch (const std::exception& ex) \
                   { __mpi_instance__.catch_call(__LINE__, __FILE__, ex); } 

#define MPIRANK __mpi_instance__.get_global_rank()
#define MPISIZE __mpi_instance__.get_global_size()

#endif//__MPIHANDLER_HPP_
*/
