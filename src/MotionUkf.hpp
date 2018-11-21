#ifndef MOTIONUKF_HPP
#define MOTIONUKF_HPP

#include "Ukf.hpp"

#include <Eigen/Dense>
#include <sstream>

class ProcessModel
{
public:
   struct MotionParams
   {
      double p_x;
      double p_y;
      double v;
      double yaw;
      double yaw_dot;
      double std_a;
      double std_yaw_dot_dot;
   };

   static inline void normalize(const long colIdx, Eigen::MatrixXd &XSigmaPointsDiff)
   {
      double & yaw = XSigmaPointsDiff.col(colIdx)(3);

      // normalize phi to -/+ pi
      yaw = yaw - TWO_PI * floor((yaw + M_PI) / TWO_PI);
   }

   static void predictSigmaPoints(const Eigen::MatrixXd & XSigmaPoints, const double d_t_s, Eigen::MatrixXd & XSigmaPointsPred);
   static inline void processModel(const MotionParams & x, const double d_t_s, MotionParams & x_pred);

   //set state dimension
   static constexpr long m_n_x = 5;

   //set augumented state dimension
   static constexpr long m_n_x_aug = 7;

   // constant 2 * PI
   static constexpr double TWO_PI = 2 * M_PI;

   static constexpr long m_nSigmaPoints = 1 + 2 * m_n_x_aug;
};


class MotionUkf : public Ukf<ProcessModel>
{
public:
   MotionUkf()
      : Ukf<ProcessModel>()
   {}

   virtual ~MotionUkf();

   struct Measurement
   {
      Measurement(int32_t dim) : value(dim) {}
      Eigen::VectorXd value;
   };


   struct RadarMeasurement : public Measurement
   {
      RadarMeasurement() : Measurement(3) {}
      RadarMeasurement(std::istringstream & iss)  : Measurement(3)
      {
         double rho;
         double phi;
         double rho_dot;
         iss >> rho;
         iss >> phi;
         iss >> rho_dot;

         value << rho, phi, rho_dot;
      }

      RadarMeasurement & operator=(std::istringstream & iss)
      {
         double rho;
         double phi;
         double rho_dot;
         iss >> rho;
         iss >> phi;
         iss >> rho_dot;

         value << rho, phi, rho_dot;
         return  *this;
      }

      double & rho() {return value(0);}
      double & phi() {return value(1);}
      double & rho_dot() {return value(2);}

      struct Definition
      {
         double rho;
         double phi;
         double rho_dot;
      };

      static Definition * cast(void * data)
      {
         return reinterpret_cast<Definition*>(data);
      }
   };

   struct RadarMeasurementModel
   {
      using MEASUREMENT_T = RadarMeasurement;

      RadarMeasurementModel();

      static inline void normalize(const long colIdx, Eigen::MatrixXd &ZSigmaPointsDiff)
      {
         //double & phi = reinterpret_cast<RadarMeasurement*>(ZSigmaPointsDiff.col(colIdx).data())->phi;
         double & phi = RadarMeasurement::cast(ZSigmaPointsDiff.col(colIdx).data())->phi;

         // normalize phi to -/+ pi
         phi = phi - TWO_PI * floor((phi + M_PI) / TWO_PI);
      }

      void predictCovar(const Ukf<ProcessModel> & ukf, const Eigen::MatrixXd & XSigmaPoints, const Eigen::VectorXd & x_pred, Eigen::MatrixXd & P_pred) const;
      void predictMeasurement(const Eigen::MatrixXd & XSigmaPointPred, Eigen::MatrixXd & ZSigmaPointsRadarPred) const;

      static constexpr long nRadarValues = 3;

      //radar measurement noise standard deviation radius in m
      const double m_std_radr = 0.3;

      //radar measurement noise standard deviation angle in rad
      const double m_std_radphi = 0.0175;

      //radar measurement noise standard deviation radius change in m/s
      const double m_std_radrd = 0.1;

      MEASUREMENT_T measurement;
   private:
      Eigen::MatrixXd m_R;
   };

   struct LaserMeasurement : public Measurement
   {
      LaserMeasurement() : Measurement(2) {}
      LaserMeasurement(std::istringstream & iss) : Measurement(2)
      {
         double px;
         double py;
         iss >> px;
         iss >> py;
         value << px , py;
      }

      LaserMeasurement & operator=(std::istringstream & iss)
      {
         double px;
         double py;
         iss >> px;
         iss >> py;
         value << px, py;
         return *this;
      }

      double & px() {return value(0);}
      double & py() {return value(1);}

      struct Definition
      {
         double px;
         double py;
      };

      static Definition * cast(void * data)
      {
         return reinterpret_cast<Definition*>(data);
      }
   };

   struct LaserMeasurementModel
   {
      using MEASUREMENT_T = LaserMeasurement;

      LaserMeasurementModel();

      static inline void normalize(const long colIdx, Eigen::MatrixXd &ZSigmaPointsDiff)
      {
         (void)colIdx;
         (void)ZSigmaPointsDiff;
         // nothing to normalize
      }

      void predictCovar(const Ukf<ProcessModel> & ukf, const Eigen::MatrixXd & XSigmaPoints, const Eigen::VectorXd & x_pred, Eigen::MatrixXd & P_pred) const;
      void predictMeasurement(const Eigen::MatrixXd & XSigmaPointPred, Eigen::MatrixXd & ZSigmaPointsLaserPred) const;

      //laser measurement noise standard deviation (position)
      const double m_std_laser_pos = 0.3;

      static constexpr long nLaserValues = 2;

      MEASUREMENT_T measurement;
   private:

      // Laser Measurement Covariance Matrix
      Eigen::MatrixXd m_L;
   };



   template<typename MEASUREMENTMODEL_T>
   void processMeasurement(const MEASUREMENTMODEL_T & measurementModel, int64_t timestamp_us)
   {
      Ukf<ProcessModel>::processMeasurement<typename MEASUREMENTMODEL_T::MEASUREMENT_T, MEASUREMENTMODEL_T>(measurementModel.measurement, measurementModel, timestamp_us);
   }

   RadarMeasurementModel radarMeasurementModel;
   LaserMeasurementModel laserMeasurementModel;
};

#endif // MOTIONUKF_HPP
