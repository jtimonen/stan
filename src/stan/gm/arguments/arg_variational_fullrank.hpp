#ifndef __STAN__GM__ARGUMENTS__VARIATIONAL__FULLRANK__HPP__
#define __STAN__GM__ARGUMENTS__VARIATIONAL__FULLRANK__HPP__

#include <stan/gm/arguments/categorical_argument.hpp>

namespace stan {

  namespace gm {

    class arg_variational_fullrank: public categorical_argument {

    public:

      arg_variational_fullrank() {

        _name = "fullrank";
        _description = "full-rank covariance";

      }

    };

  } // gm

} // stan

#endif

