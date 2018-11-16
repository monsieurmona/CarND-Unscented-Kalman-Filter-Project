#ifndef MOTIONUKF_HPP
#define MOTIONUKF_HPP

#include "Ukf.hpp"

#include <Eigen/Dense>

class MotionUkf : public Ukf
{
public:
   MotionUkf();
   virtual ~MotionUkf();

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

   struct RadarMeasurement
   {
      double rho;
      double phi;
      double rho_dot;
   };

   class ProcessModel {
   public:
      static inline void normalize(const long colIdx, Eigen::MatrixXd &XSigmaPointsDiff)
      {
         double & yaw = XSigmaPointsDiff.col(colIdx)(3);

         // normalize phi to -/+ pi
         // TODO move down
         yaw = yaw - TWO_PI * floor((yaw + M_PI) / TWO_PI);
      }
   };

   class RadarMeasurementModel {
   public:
      static inline void normalize(const long colIdx, Eigen::MatrixXd &ZSigmaPointsDiff)
      {
         double & phi = reinterpret_cast<RadarMeasurement*>(ZSigmaPointsDiff.col(colIdx).data())->phi;

         // normalize phi to -/+ pi
         // TODO move down
         phi = phi - TWO_PI * floor((phi + M_PI) / TWO_PI);
      }
   };

   inline void predict(const Eigen::MatrixXd & XSigmaPoints, Eigen::VectorXd &x_pred, Eigen::MatrixXd & P_pred)
   {
      P_pred.setZero(m_n_x, m_n_x);
      Ukf::predict<ProcessModel>(XSigmaPoints, x_pred, P_pred);
   }

   virtual void predictSigmaPoints(const Eigen::MatrixXd & XSigmaPoints, const double d_t_s, Eigen::MatrixXd & XSigmaPointsPred);

   void predictRadarCovar(const Eigen::MatrixXd & XSigmaPoints, const Eigen::VectorXd & x_pred, Eigen::MatrixXd & P_pred);

   inline void processModel(const MotionParams & x, const double d_t_s, MotionParams & x_pred);

   void predictRadarMeasurement(const Eigen::MatrixXd & XSigmaPointPred, Eigen::MatrixXd & ZSigmaPointsRadarPred);

   const long nRadarValues = 3;
private:

   //radar measurement noise standard deviation radius in m
   const double m_std_radr = 0.3;

   //radar measurement noise standard deviation angle in rad
   const double m_std_radphi = 0.0175;

   //radar measurement noise standard deviation radius change in m/s
   const double m_std_radrd = 0.1;

   Eigen::MatrixXd m_R;
};

#endif // MOTIONUKF_HPP
