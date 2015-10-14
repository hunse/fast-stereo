
/* -----------------------------------------------------------------------
 * A header with simple functions for timing on Linux with microsecond
 * accuracy (assuming the processor is using this level of accuracy)
 * -----------------------------------------------------------------------
 */

#ifndef _1C66A83A_8A68_4AC4_B5C5_DCB705F05BE5_H_
#define _1C66A83A_8A68_4AC4_B5C5_DCB705F05BE5_H_

#include <stdio.h>
#include <stdint.h>
/* #include <time.h> */
#include <sys/time.h>

#include <list>
#include <string>

typedef uint64_t us_t;

us_t now()
// Return the current time in microseconds
{
	timeval t;
	gettimeofday(&t, NULL);
	return 1000000 * t.tv_sec + t.tv_usec;
}

us_t getElapsedMicroseconds(us_t start)
{
    return now() - start;
}
double getElapsedMilliseconds(us_t start)
{
    return (1.0E-3) * getElapsedMicroseconds(start);
}
double getElapsedSeconds(us_t start)
{
    return (1.0E-6) * getElapsedMicroseconds(start);
}

void printElapsedMicroseconds(us_t start)
{
    printf("%10lu us elapsed\n", getElapsedMicroseconds(start));
}
void printElapsedMilliseconds(us_t start)
{
    printf("%10.3f ms elapsed\n", getElapsedMilliseconds(start));
}
void printElapsedSeconds(us_t start)
{
    printf("%10.3f s elapsed\n", getElapsedSeconds(start));
}

// Matlab-like tic and toc functions for easy timing
class TimeTag
{
public:
    std::string tag;
    us_t time;
    TimeTag(us_t time_, std::string tag_="") : time(time_), tag(tag_) {}
};

std::list<TimeTag> timers;

void tic(std::string tag = "")
{
    TimeTag timer(now(), tag);
    timers.push_back(timer);
}

void toc()
{
    TimeTag timer = timers.back();
    double seconds = getElapsedSeconds(timer.time);
    timers.pop_back();

    if (!timer.tag.empty())
        std::cout << timer.tag << ": ";
    printf("%0.3f s elapsed\n", seconds);
}

#endif
