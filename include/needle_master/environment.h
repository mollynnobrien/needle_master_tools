#include <boost/polygon/polygon.hpp>

namespace gtl = boost::polygon;

class Gate;
class Surface;

class Environment {
  double height;
  double width;
};

class Gate {
private:
    gtl::Polygon box;
    gtl::Polygon top;
    gtl::Polygon bottom;
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
