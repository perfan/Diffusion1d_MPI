// 
// sparkline.cc
//
// write 1d rarray as a sparkline

#include "sparkline.h"
#include <algorithm>
#include <limits>
#include <cmath>

template<class T>
static std::string sparkline_g(const rvector<T>& x, int width,
                               int  nskipleft, int  nskipright,
                               T x1 = std::numeric_limits<T>::lowest(), T x2 = std::numeric_limits<T>::max())
{
    // sparkline_g is a template function produces a one-line string
    // with a graph made up of (unicode) block characters whose height
    // represents the value of the array.  For long arrays, it allows
    // a course-graining by specifying the number of characters to
    // produce, where each character covers a part of the array. The
    // height of each character is determined from the maximum of the
    // values within that part. There's an option for specifying
    // whether the bottom of the characters should represent zero.
    //
    // This is a generic function that is used in the implementation
    // of the sparkline functions for char, short, int, long, long
    // long, their unsigned variants, and the floating point types
    // float, double and long double.
    //
    // input parameters:
    //  x:      rvector of values
    //  width:  width of the graph (int)
    //  x1:     lower bound of values
    //  x2:     upper bound of values
    // 
    // returns: the sparkline string in utf-8 format.
    //
    int per = (width<1)?1:((x.size()+width-1)/width);
    std::string result(3*((x.size()+per-1)/per), ' ');
    auto minmaxval = std::minmax_element(&x[0], &x[0] + x.size());
    T min, max;
    if (x1 != std::numeric_limits<T>::lowest())
        min = x1;
    else
        min = *minmaxval.first;
    if (x2 != std::numeric_limits<T>::max())
        max = x2;
    else
        max = *minmaxval.second;
    T range = max - min;
    int i = 0;
    const unsigned char sparkchars[9] = {129,130,131,132,133,134,135,136,171};
    for (int j=0;j<x.size();j+=per) {
        if (per > x.size()-j) per = x.size()-j;
        T value = *std::max_element(&x[j], &x[j] + per);
        if (!std::isnan(value)) {
            int level = (int)round(7*(value-min)/range);
            if (level>=0 and level<8) {
                result[i++] = 226;
                result[i++] = 150;
                result[i++] = sparkchars[level];
            } else {
                result[i++] = 226;
                result[i++] = 150;
                result[i++] = sparkchars[8];
            }
        } else {
            result[i++] = 226;
            result[i++] = 150;
            result[i++] = sparkchars[8];
        }
    }
    return result;
}

#ifndef NOMPISPARKLINE
#include <mpi.h>
template<class T>
static std::string mpi_sparkline_g(const rvector<T>& x,
                                   int          width,
                                   MPI_Datatype mpitype,
                                   int          root,
                                   MPI_Comm     comm,
                                   int          nskipleft=0,
                                   int          nskipright=0,
                                   T            x1 = std::numeric_limits<T>::lowest(),
                                   T            x2 = std::numeric_limits<T>::max())
{
    long myn = x.size();
    long totn;
    MPI_Allreduce(&myn, &totn, 1, MPI_LONG, MPI_SUM, comm);
    int mywidth = (width * float(myn))/(totn) + 0.5;
    T mymin, mymax;
    auto minmaxval = std::minmax_element(&x[0], &x[0] + x.size());
    if (x1 != std::numeric_limits<T>::lowest())
        mymin = x1;
    else
        mymin = *minmaxval.first;
    if (x2 != std::numeric_limits<T>::max())
        mymax = x2;
    else
        mymax = *minmaxval.second;
    T totmin, totmax;
    MPI_Allreduce(&mymin, &totmin, 1, mpitype, MPI_MIN, comm);
    MPI_Allreduce(&mymax, &totmax, 1, mpitype, MPI_MAX, comm);
    if (x1 != std::numeric_limits<T>::lowest())
        mymin = x1;
    else
        mymin = totmin;
    if (x2 != std::numeric_limits<T>::max())
        mymax = x2;
    else
        mymax = totmax;
    std::string mysparklinestr = sparkline_g(x, mywidth, nskipleft, nskipright, mymin, mymax);
    long mysparklinestrlen = mysparklinestr.size();
    int rank, commsize;
    MPI_Comm_size(comm, &commsize);
    MPI_Comm_rank(comm, &rank);
    long allsparklinestrlen[commsize];
    MPI_Gather(&mysparklinestrlen,1,MPI_LONG,allsparklinestrlen,1,MPI_LONG,root,comm);
    const int tag = 15;
    if (rank == root) {
        std::vector<std::string> allsparklinestr(commsize);
        for (int r = 0; r < commsize; r++) {
            allsparklinestr[r] = std::string(allsparklinestrlen[r],' ');
            if (r != root)
                MPI_Recv(&(allsparklinestr[r][0]), allsparklinestrlen[r], MPI_CHAR, r, tag, comm, MPI_STATUS_IGNORE);
            else
                allsparklinestr[r] = mysparklinestr;
        }
        std::string result;
        for (int r = 0; r < commsize; r++)
            result.append(allsparklinestr[r]);
        return result;
    } else {
        MPI_Ssend(mysparklinestr.data(), mysparklinestrlen, MPI_CHAR, root, tag, comm);
        return {""};
    }
}
#endif

// zero:   whether the bottom of the characters should represent zero (bool)
template<class T>
static std::string sparkline_g(const rvector<T>& x, int width, int nskipleft, int nskipright, bool zero)
{
    return sparkline_g(x, width, nskipleft, nskipright, zero?(T(0)):(std::numeric_limits<T>::lowest()), std::numeric_limits<T>::max());
}

template<class T>
static std::string mpi_sparkline_g(const rvector<T>& x, int width, MPI_Datatype mpitype, int root, MPI_Comm comm, int nskipleft, int nskipright, bool zero)
{
    return mpi_sparkline_g(x, width, mpitype, root, comm, nskipleft, nskipright, zero?(T(0)):(std::numeric_limits<T>::lowest()), std::numeric_limits<T>::max());
}

template<class T>
static std::string sparkhist_g(T x1, T x2, const rvector<T>& x, int width, bool zero)
{
    if (width<1) width = 21;
    rvector<unsigned int> hist(width);

    T deltax = (x2-x1)/width;
    if (deltax==T(0)) deltax=T(1);
    hist.fill(0);
    for (auto& xvalue: x)
        hist[(unsigned int)((xvalue-x1)/deltax)]++;
    return sparkline_g(hist, width, 0, 0, zero);
}

template<class T>
static std::string sparkhist_g(const rvector<T>& x, int width, bool zero)
{
    auto minmaxval = std::minmax_element(&x[0], &x[0] + x.size());
    T min = *minmaxval.first;
    T max = *minmaxval.second;
    T x1 = min, x2;
    if (max < T(0)) {
        x2 = max*(1-2*std::numeric_limits<T>::epsilon()) + std::numeric_limits<T>::is_integer;
    } else {
        x2 = max*(1+2*std::numeric_limits<T>::epsilon()) + std::numeric_limits<T>::is_integer;
    }
    return sparkhist_g(x1, x2, x, width, zero);
}

// sparkline implementation, first form

std::string sparkline(const rvector<char>& x,               int width, int nskipleft, int nskipright, bool zero) { return sparkline_g(x, width, nskipleft, nskipright, zero); }
std::string sparkline(const rvector<short>& x,              int width, int nskipleft, int nskipright, bool zero) { return sparkline_g(x, width, nskipleft, nskipright, zero); }
std::string sparkline(const rvector<int>& x,                int width, int nskipleft, int nskipright, bool zero) { return sparkline_g(x, width, nskipleft, nskipright, zero); }
std::string sparkline(const rvector<long>& x,               int width, int nskipleft, int nskipright, bool zero) { return sparkline_g(x, width, nskipleft, nskipright, zero); }
std::string sparkline(const rvector<long long>& x,          int width, int nskipleft, int nskipright, bool zero) { return sparkline_g(x, width, nskipleft, nskipright, zero); }
std::string sparkline(const rvector<unsigned char>& x,      int width, int nskipleft, int nskipright, bool zero) { return sparkline_g(x, width, nskipleft, nskipright, zero); }
std::string sparkline(const rvector<unsigned short>& x,     int width, int nskipleft, int nskipright, bool zero) { return sparkline_g(x, width, nskipleft, nskipright, zero); }
std::string sparkline(const rvector<unsigned int>& x,       int width, int nskipleft, int nskipright, bool zero) { return sparkline_g(x, width, nskipleft, nskipright, zero); }
std::string sparkline(const rvector<unsigned long>& x,      int width, int nskipleft, int nskipright, bool zero) { return sparkline_g(x, width, nskipleft, nskipright, zero); }
std::string sparkline(const rvector<unsigned long long>& x, int width, int nskipleft, int nskipright, bool zero) { return sparkline_g(x, width, nskipleft, nskipright, zero); }
std::string sparkline(const rvector<float>& x,              int width, int nskipleft, int nskipright, bool zero) { return sparkline_g(x, width, nskipleft, nskipright, zero); }
std::string sparkline(const rvector<double>& x,             int width, int nskipleft, int nskipright, bool zero) { return sparkline_g(x, width, nskipleft, nskipright, zero); }
std::string sparkline(const rvector<long double>& x,        int width, int nskipleft, int nskipright, bool zero) { return sparkline_g(x, width, nskipleft, nskipright, zero); }

std::string mpi_sparkline(const rvector<double>& x,             int width, int root, MPI_Comm comm, int nskipleft, int nskipright, bool zero) { return mpi_sparkline_g(x, width, MPI_DOUBLE, root, comm, nskipleft, nskipright, zero); }

// sparkline implementation, second form

std::string sparkline(const rvector<char>& x,               int width, char x1,               char x2              ) { return sparkline_g(x, width, x1, x2); }
std::string sparkline(const rvector<short>& x,              int width, short x1,              short x2             ) { return sparkline_g(x, width, x1, x2); }
std::string sparkline(const rvector<int>& x,                int width, int x1,                int x2               ) { return sparkline_g(x, width, x1, x2); }
std::string sparkline(const rvector<long>& x,               int width, long x1,               long x2              ) { return sparkline_g(x, width, x1, x2); }
std::string sparkline(const rvector<long long>& x,          int width, long long x1,          long long x2         ) { return sparkline_g(x, width, x1, x2); }
std::string sparkline(const rvector<unsigned char>& x,      int width, unsigned char x1,      unsigned char x2     ) { return sparkline_g(x, width, x1, x2); }
std::string sparkline(const rvector<unsigned short>& x,     int width, unsigned short x1,     unsigned short x2    ) { return sparkline_g(x, width, x1, x2); }
std::string sparkline(const rvector<unsigned int>& x,       int width, unsigned int x1,       unsigned int x2      ) { return sparkline_g(x, width, x1, x2); }
std::string sparkline(const rvector<unsigned long>& x,      int width, unsigned long x1,      unsigned long x2     ) { return sparkline_g(x, width, x1, x2); }
std::string sparkline(const rvector<unsigned long long>& x, int width, unsigned long long x1, unsigned long long x2) { return sparkline_g(x, width, x1, x2); }
std::string sparkline(const rvector<float>& x,              int width, float x1,              float x2             ) { return sparkline_g(x, width, x1, x2); }
std::string sparkline(const rvector<double>& x,             int width, double x1,             double x2            ) { return sparkline_g(x, width, x1, x2); }
std::string sparkline(const rvector<long double>& x,        int width, long double x1,        long double x2       ) { return sparkline_g(x, width, x1, x2); }

// sparkhist implementation, first form

std::string sparkhist(const rvector<char>& x, int width, bool zero)               { return sparkhist_g(x, width, zero); }
std::string sparkhist(const rvector<short>& x, int width, bool zero)              { return sparkhist_g(x, width, zero); }
std::string sparkhist(const rvector<int>& x, int width, bool zero)                { return sparkhist_g(x, width, zero); }
std::string sparkhist(const rvector<long>& x, int width, bool zero)               { return sparkhist_g(x, width, zero); }
std::string sparkhist(const rvector<long long>& x, int width, bool zero)          { return sparkhist_g(x, width, zero); }
std::string sparkhist(const rvector<unsigned char>& x, int width, bool zero)      { return sparkhist_g(x, width, zero); }
std::string sparkhist(const rvector<unsigned short>& x, int width, bool zero)     { return sparkhist_g(x, width, zero); }
std::string sparkhist(const rvector<unsigned int>& x, int width, bool zero)       { return sparkhist_g(x, width, zero); }
std::string sparkhist(const rvector<unsigned long>& x, int width, bool zero)      { return sparkhist_g(x, width, zero); }
std::string sparkhist(const rvector<unsigned long long>& x, int width, bool zero) { return sparkhist_g(x, width, zero); }
std::string sparkhist(const rvector<float>& x, int width, bool zero)              { return sparkhist_g(x, width, zero); }
std::string sparkhist(const rvector<double>& x, int width, bool zero)             { return sparkhist_g(x, width, zero); }
std::string sparkhist(const rvector<long double>& x, int width, bool zero)        { return sparkhist_g(x, width, zero); }

// sparkhist implementation, second form

std::string sparkhist(char x1, char x2, const rvector<char>& x, int width, bool zero)                                           { return sparkhist_g(x1,x2,x,width,zero); }
std::string sparkhist(short x1, short x2, const rvector<short>& x, int width, bool zero)                                        { return sparkhist_g(x1,x2,x,width,zero); }
std::string sparkhist(int x1, int x2, const rvector<int>& x, int width, bool zero)                                              { return sparkhist_g(x1,x2,x,width,zero); }
std::string sparkhist(long x1, long x2, const rvector<long>& x, int width, bool zero)                                           { return sparkhist_g(x1,x2,x,width,zero); }
std::string sparkhist(long long x1, long long x2, const rvector<long long>& x, int width, bool zero)                            { return sparkhist_g(x1,x2,x,width,zero); }
std::string sparkhist(unsigned char x1, unsigned char x2, const rvector<unsigned char>& x, int width, bool zero)                { return sparkhist_g(x1,x2,x,width,zero); }
std::string sparkhist(unsigned short x1, unsigned short x2, const rvector<unsigned short>& x, int width, bool zero)             { return sparkhist_g(x1,x2,x,width,zero); }
std::string sparkhist(unsigned int x1, unsigned int x2, const rvector<unsigned int>& x, int width, bool zero)                   { return sparkhist_g(x1,x2,x,width,zero); }
std::string sparkhist(unsigned long x1, unsigned long x2, const rvector<unsigned long>& x, int width, bool zero)                { return sparkhist_g(x1,x2,x,width,zero); }
std::string sparkhist(unsigned long long x1, unsigned long long x2, const rvector<unsigned long long>& x, int width, bool zero) { return sparkhist_g(x1,x2,x,width,zero); }
std::string sparkhist(float x1, float x2, const rvector<float>& x, int width, bool zero)                                        { return sparkhist_g(x1,x2,x,width,zero); }
std::string sparkhist(double x1, double x2, const rvector<double>& x, int width, bool zero)                                     { return sparkhist_g(x1,x2,x,width,zero); }
std::string sparkhist(long double x1, long double x2, const rvector<long double>& x, int width, bool zero)                      { return sparkhist_g(x1,x2,x,width,zero); }

// done
