#ifndef MEASUREMENTPACKAGE_HPP
#define MEASUREMENTPACKAGE_HPP

#include "Eigen/Dense"

class MeasurementPackage {
public:
    int64_t timestamp_;

    enum SensorType {
        LASER, RADAR
    } sensor_type_;

    Eigen::VectorXd raw_measurements;
};

#endif // MEASUREMENTPACKAGE_HPP
