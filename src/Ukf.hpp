#ifndef UKF_HPP
#define UKF_HPP

#include <Eigen/Dense>

class UkfBase
{

};

template<typename PROCESSMODEL_T>
class Ukf
{
public:
   Ukf()
      : m_n_x(PROCESSMODEL_T::m_n_x)
      , m_n_x_aug(PROCESSMODEL_T::m_n_x_aug)
      , m_lambda(3 - PROCESSMODEL_T::m_n_x_aug)
      , m_weights(calculateWeights())
      , m_x(m_n_x)
      , m_P(m_n_x, m_n_x)
      , m_previousTimestampUs(0)
   {
   }

   ~Ukf() {}

   // delete this as this class as an read only interface
   Ukf(const Ukf & rhs) = delete;
   Ukf & operator=(Ukf & ukf) = delete;
   Ukf & operator=(Ukf && ukf) = delete;


   template<typename MEASUREMENT_T, typename MEASUREMENTMODEL_T>
   void processMeasurement(const MEASUREMENT_T & z_measurement, const MEASUREMENTMODEL_T & measurementModel, int64_t timestamp_us)
   {
      const int64_t dt_us = setNewTime(timestamp_us);

      if (dt_us > 0)
      {
         // do update step
         // if some time passed since last measurement
         const double dt_s = dt_us / 1000000.0;	//dt - expressed in seconds

         // generate augumented sigma points from last state

         // TODO find out if sigma points must be generated again
         // it might be enough to use the predicted sigma points
         // from last round
         Eigen::MatrixXd XSigmaPoints;
         generateAugumentedSigmaPoints(m_x, m_P, m_std, XSigmaPoints);

         // predict sigma points
         PROCESSMODEL_T::predictSigmaPoints(XSigmaPoints, dt_s, m_XSigmaPointsPred);

         // predict object covariance and
         // predict object state
         m_x_pred.setZero(PROCESSMODEL_T::m_n_x);
         m_P_pred.setZero(PROCESSMODEL_T::m_n_x, PROCESSMODEL_T::m_n_x);

         predict<PROCESSMODEL_T>(XSigmaPoints, m_x_pred, m_P_pred);
      }

      Eigen::MatrixXd Z_pred;
      measurementModel.predictMeasurement(m_XSigmaPointsPred, Z_pred);

      Eigen::VectorXd z_pred;
      weightedMean(Z_pred, z_pred);

      Eigen::MatrixXd S_cov;
      measurementModel.predictCovar(*this, Z_pred, z_pred, S_cov);

      //
      // update
      //


      // Cross Correlation Matrix
      Eigen::MatrixXd Tc;
      calculateCrosscorrelationMatix(m_XSigmaPointsPred, Z_pred, m_x_pred, z_pred, Tc);

      // Kalman Gain
      Eigen::MatrixXd K;
      calculateKalmanGain(Tc, S_cov, K);

      // update state and covariance matrix
      updateState(K, z_measurement, z_pred, m_x_pred);
      updateCovariance(K, S_cov, m_P_pred);

      std::swap(m_x_pred, m_x);
      std::swap(m_P_pred, m_P);
   }

   template<typename MODEL_T>
   void predictCovar(const Eigen::MatrixXd & SigmaPoints, const Eigen::VectorXd & vector, Eigen::MatrixXd & Covariance) const
   {
      Eigen::MatrixXd SigmaPointsDiff = SigmaPoints.colwise() - vector;

      for (long colIdx = 0; colIdx < PROCESSMODEL_T::m_nSigmaPoints; ++colIdx)
      {
         // TODO move down
         MODEL_T::normalize(colIdx, SigmaPointsDiff);

         Eigen::MatrixXd cov_pred_sigmaPoint = (SigmaPointsDiff.col(colIdx).array() * m_weights(colIdx)).matrix() *
               SigmaPointsDiff.col(colIdx).transpose();

         Covariance += cov_pred_sigmaPoint;
      }
   }

   // set standard deviation - process noise
   void setStd(const Eigen::VectorXd & std) { m_std = std; }

   // constant 2 * PI
   static constexpr double TWO_PI = 2 * M_PI;

private:
   //set state dimension
   const long m_n_x;

   //set augumented state dimension
   const long m_n_x_aug;

public:
   // Read only interface
   const Eigen::VectorXd & x = m_x;
   const Eigen::MatrixXd & P = m_P;

protected:
   void generateSigmaPoints(const Eigen::VectorXd &x_, const Eigen::MatrixXd & P_, Eigen::MatrixXd & XSigmaPoints)
   {
      //create sigma point matrix
      XSigmaPoints.setZero(m_n_x, 2 * m_n_x + 1);

      //calculate square root of P
      Eigen::MatrixXd A = P_.llt().matrixL();

      //set first column of sigma point matrix
      XSigmaPoints.col(0)  = x_;

      //set remaining sigma points
      for (int i = 0; i < m_n_x; i++)
      {
         XSigmaPoints.col(i+1)       = x_ + sqrt(m_lambda+m_n_x) * A.col(i);
         XSigmaPoints.col(i+1+m_n_x) = x_ - sqrt(m_lambda+m_n_x) * A.col(i);
      }
   }


   void generateAugumentedSigmaPoints(const Eigen::VectorXd &x_, const Eigen::MatrixXd & P_, const Eigen::VectorXd &std, Eigen::MatrixXd & XSigmaPoints)
   {
      //create augmented mean vector
      Eigen::VectorXd x_aug;
      x_aug.setZero(m_n_x_aug);
      x_aug.head(m_n_x) = x_;

      //create augmented state covariance
      Eigen::MatrixXd P_aug;
      P_aug.setZero(m_n_x_aug, m_n_x_aug);
      P_aug.topLeftCorner(m_n_x, m_n_x) = P_;

      for (long i = m_n_x; i < m_n_x_aug; ++i)
      {
         P_aug(i, i) = pow(std(i - m_n_x),2);
      }

      //create sigma point matrix
      XSigmaPoints.setZero(m_n_x_aug, 2 * m_n_x_aug + 1);

      //calculate square root of P
      Eigen::MatrixXd A = P_aug.llt().matrixL();

      //set first column of sigma point matrix
      XSigmaPoints.col(0)  = x_aug;

      //set remaining sigma points
      for (int i = 0; i < m_n_x_aug; i++)
      {
         XSigmaPoints.col(i+1)           = x_aug + sqrt(m_lambda+m_n_x_aug) * A.col(i);
         XSigmaPoints.col(i+1+m_n_x_aug) = x_aug - sqrt(m_lambda+m_n_x_aug) * A.col(i);
      }
   }


   void weightedMean(const Eigen::MatrixXd & XSigmaPoints, Eigen::VectorXd &x_pred)
   {
      x_pred = (XSigmaPoints.array().rowwise() * m_weights.transpose()).rowwise().sum();
   }


   void calculateCrosscorrelationMatix(const Eigen::MatrixXd & XSigmaPointsPred, const Eigen::MatrixXd & ZSigmaPointsPred, const Eigen::VectorXd & x_mean, const Eigen::VectorXd & z_mean, Eigen::MatrixXd & T)
   {
      const Eigen::MatrixXd X = (XSigmaPointsPred.colwise() - x_mean);
      const Eigen::MatrixXd Z = (ZSigmaPointsPred.colwise() - z_mean);
      const Eigen::MatrixXd X_weighted = X.array().rowwise() * m_weights.transpose();
      T = (X_weighted * Z.transpose());
   }

   void calculateKalmanGain(const Eigen::MatrixXd & T, const Eigen::MatrixXd & S, Eigen::MatrixXd & K)
   {
      K = T * S.inverse();
   }

   template <typename MEASUREMENT_T>
   //void updateState(const Eigen::MatrixXd & K, const Eigen::VectorXd & z, const Eigen::VectorXd & z_pred, Eigen::VectorXd & x_pred)
   void updateState(const Eigen::MatrixXd & K, const MEASUREMENT_T & z, const Eigen::VectorXd & z_pred, Eigen::VectorXd & x_pred)
   {
      Eigen::VectorXd & x_update = x_pred;
      x_update = x_pred + K * (z.value - z_pred);
   }

   void updateCovariance(const Eigen::MatrixXd & K, const Eigen::MatrixXd & S, Eigen::MatrixXd & P_pred) const
   {
      Eigen::MatrixXd & P_update = P_pred;
      P_update = P_pred - K * S * K.transpose();
   }

   template<typename MODEL_T>
   inline void predict(const Eigen::MatrixXd & XSigmaPoints, Eigen::VectorXd &x_pred, Eigen::MatrixXd & P_pred)
   {
      weightedMean(XSigmaPoints, x_pred);
      predictCovar<MODEL_T>(XSigmaPoints, x_pred, P_pred);
   }


   // set the new measurement time and compute the difference
   inline int64_t setNewTime(const int64_t timestamp_us)
   {
      //compute the time elapsed between the current and previous measurements
      int64_t dt = timestamp_us - m_previousTimestampUs;
      m_previousTimestampUs = timestamp_us;
      return dt;
   }


   //define spreading parameter
   const double m_lambda;

   // weights for prediction
   const Eigen::ArrayXd m_weights;


   // set standard deviation - process noise
   Eigen::Vector2d m_std;


   // object state
   Eigen::VectorXd m_x;

   // object covariance matrix
   Eigen::MatrixXd m_P;

   // predicted object state
   Eigen::VectorXd m_x_pred;

   // predicted object covariance matrix
   Eigen::MatrixXd m_P_pred;

   // predicted Sigma Points
   Eigen::MatrixXd m_XSigmaPointsPred;

private:
   const Eigen::ArrayXd calculateWeights()
   {
      Eigen::ArrayXd weights(PROCESSMODEL_T::m_nSigmaPoints);

      weights(0) = m_lambda / (m_lambda + m_n_x_aug);

      for (int i = 1; i < PROCESSMODEL_T::m_nSigmaPoints; ++i)
      {
         weights(i) = 0.5 / (m_lambda + m_n_x_aug);
      }

      return  weights;
   }

   int64_t m_previousTimestampUs;
};

#endif // UKF_HPP
