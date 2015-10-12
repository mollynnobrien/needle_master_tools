#ifndef _NM_TOOLS
#define _NM_TOOLS

// Boost polygons
#include <boost/polygon/polygon.hpp>

// Boost python
#include <boost/python.hpp>
#include <boost/python/list.hpp>

// Boost threads
#include <boost/thread/mutex.hpp>

// polygon stuff based off of the Boost examples
// e.g.:
// http://www.boost.org/doc/libs/1_46_1/libs/polygon/doc/gtl_polygon_usage.htm
namespace gtl = boost::polygon;

namespace needle_master {

  typedef gtl::polygon_data<int> Polygon;
  typedef gtl::polygon_traits<Polygon>::point_type Point;

  class Gate;
  class Surface;

  class Environment {
    double height;
    double width;

    std::vector<Gate> gates;

    std::vector<Surface> surfaces;

  public:

    Environment(double height, double width);

    /**
     * Python wrapper for adding a gate to an environment
     */
    int pyAddGate(double x, double y, double w,
                   const boost::python::list &pts,
                   const boost::python::list &top_pts,
                   const boost::python::list &bottom_pts);

    /**
     * Python wrapper for adding a surface to the environment
     */
    int pyAddSurface(const boost::python::list &pts, bool isDeepTissue);

  };

  class Gate {
  private:
    Polygon box;
    Polygon top;
    Polygon bottom;
    double x;
    double y;
    double w;
    double height;
    double width;
  };

  class Surface {
  private:
    bool isDeepTissue;

  public:

  };
}

#endif
