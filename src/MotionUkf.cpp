#include "MotionUkf.hpp"

#include <math.h>


MotionUkf::~MotionUkf() {}

void ProcessModel::predictSigmaPoints(const Eigen::MatrixXd & XSigmaPoints, const double d_t_s, Eigen::MatrixXd & XSigmaPointsPred)
{
   XSigmaPointsPred.setZero(m_n_x, 2 * m_n_x_aug + 1);

   for (int colIdx = 0; colIdx < XSigmaPoints.cols(); ++colIdx)
   {
      MotionParams & x_pred = *reinterpret_cast<MotionParams*>(XSigmaPointsPred.col(colIdx).data());
      const MotionParams & x = *reinterpret_cast<const MotionParams * const>(XSigmaPoints.col(colIdx).data());
      processModel(x, d_t_s, x_pred);
   }
}

inline void ProcessModel::processModel(const MotionParams & x, const double d_t_s, MotionParams & x_pred)
{
   // just for better readability
   const double one_half = 0.5;
   const double one_half_dt2 = one_half * pow(d_t_s, 2);

   // predict position
   if (fabs(x.yaw_dot) > 0.001)
   {
      x_pred.p_x = x.p_x + (x.v / x.yaw_dot) * (sin(x.yaw + x.yaw_dot * d_t_s) - sin(x.yaw));
      x_pred.p_y = x.p_y + (x.v / x.yaw_dot) * (cos(x.yaw) - cos(x.yaw + x.yaw_dot * d_t_s));
   }
   else
   {
      x_pred.p_x = x.p_x + (x.v * cos(x.yaw) * d_t_s);
      x_pred.p_y = x.p_y + (x.v * sin(x.yaw) * d_t_s);
   }

   // add noise to position
   x_pred.p_x += one_half_dt2 * cos(x.yaw) * x.std_a;
   x_pred.p_y += one_half_dt2 * sin(x.yaw) * x.std_a;

   // velocity
   x_pred.v = x.v + 0.0 + // speed change
         d_t_s * x.std_a; // noise

   // yaw
   x_pred.yaw = x.yaw + x.yaw_dot * d_t_s + // yaw change
         one_half_dt2  * x.std_yaw_dot_dot; // noise

   // yaw_dot
   x_pred.yaw_dot = x.yaw_dot + 0.0 + // yaw_dot change
         d_t_s * x.std_yaw_dot_dot; // noise
}

constexpr long MotionUkf::RadarMeasurementModel::nRadarValues;

MotionUkf::RadarMeasurementModel::RadarMeasurementModel() : m_R(nRadarValues,nRadarValues)
{
   m_R << pow(m_std_radr, 2), 0,                    0,
          0,                  pow(m_std_radphi, 2), 0,
          0,                  0,                    pow(m_std_radrd, 2);
}


void MotionUkf::RadarMeasurementModel::predictCovar(const Ukf<ProcessModel> & ukf, const Eigen::MatrixXd & ZSigmaPoints, const Eigen::VectorXd & z_mean, Eigen::MatrixXd & S_pred) const
{
   S_pred.setZero(nRadarValues, nRadarValues);
   ukf.predictCovar<MotionUkf::RadarMeasurementModel>(ZSigmaPoints, z_mean, S_pred);
   S_pred += m_R;
}


void MotionUkf::RadarMeasurementModel::predictMeasurement(const Eigen::MatrixXd & XSigmaPointPred, Eigen::MatrixXd & ZSigmaPointsRadarPred) const
{
   const long n_z = nRadarValues;
   ZSigmaPointsRadarPred.setZero(n_z, ProcessModel::m_nSigmaPoints);

   for (long colIdx = 0; colIdx < ProcessModel::m_nSigmaPoints; ++colIdx)
   {
      const ProcessModel::MotionParams & x_sig = *reinterpret_cast<const ProcessModel::MotionParams*>(XSigmaPointPred.col(colIdx).data());
      //RadarMeasurement & z = *reinterpret_cast<RadarMeasurement*>(ZSigmaPointsRadarPred.col(colIdx).data());
      RadarMeasurement::Definition & z = *RadarMeasurement::cast(ZSigmaPointsRadarPred.col(colIdx).data());

      z.rho = sqrt(pow(x_sig.p_x, 2) + pow(x_sig.p_y, 2));

      if (fabs(z.rho) > 0.0001)
      {
         z.phi = atan2(x_sig.p_y, x_sig.p_x);
         z.rho_dot = (x_sig.p_x * cos(x_sig.yaw) * x_sig.v + x_sig.p_y * sin(x_sig.yaw) * x_sig.v) / z.rho;
      }
   }
}

constexpr long MotionUkf::LaserMeasurementModel::nLaserValues;

MotionUkf::LaserMeasurementModel::LaserMeasurementModel() : m_L(nLaserValues,nLaserValues)
{
   m_L << pow(m_std_laser_pos, 2), 0,
          0,                  pow(m_std_laser_pos, 2);
}

void MotionUkf::LaserMeasurementModel::predictCovar(const Ukf<ProcessModel> & ukf, const Eigen::MatrixXd & ZSigmaPoints, const Eigen::VectorXd & z_mean, Eigen::MatrixXd & S_pred) const
{
   S_pred.setZero(nLaserValues, nLaserValues);
   ukf.predictCovar<MotionUkf::RadarMeasurementModel>(ZSigmaPoints, z_mean, S_pred);
   S_pred += m_L;
}


void MotionUkf::LaserMeasurementModel::predictMeasurement(const Eigen::MatrixXd & XSigmaPointPred, Eigen::MatrixXd & ZSigmaPointsLaserPred) const
{
   ZSigmaPointsLaserPred = XSigmaPointPred.block<nLaserValues, ProcessModel::m_nSigmaPoints>(0,0);
}

