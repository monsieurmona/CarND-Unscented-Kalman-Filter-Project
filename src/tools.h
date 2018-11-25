#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools {
public:
  /**
  * Constructor.
  */
  Tools() = delete;

  /**
  * A helper method to calculate RMSE.
  */
  static VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  // normalize angle to -/+ pi
  static inline void normalizeAngle(double & angle)
  {
     angle = angle - TWO_PI * floor((angle + M_PI) / TWO_PI);
  }

  // constant 2 * PI
  static constexpr double TWO_PI = 2 * M_PI;
};

#endif /* TOOLS_H_ */
