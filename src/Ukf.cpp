#include "Ukf.hpp"

Ukf::Ukf(const long n_x)
   : m_n_x(n_x)
   , m_n_x_aug(n_x)
   , m_lambda(3 - n_x)
   , m_nSigmaPoints(1 + 2 * m_n_x_aug)
   , m_weights(calculateWeights())
{

}

Ukf::Ukf(const long n_x, const long n_x_aug)
   : m_n_x(n_x)
   , m_n_x_aug(n_x_aug)
   , m_lambda(3 - n_x_aug)
   , m_nSigmaPoints(1 + 2 * m_n_x_aug)
   , m_weights(calculateWeights())
{

}

Ukf::~Ukf() {}


void Ukf::generateSigmaPoints(const Eigen::VectorXd &x, const Eigen::MatrixXd & P, Eigen::MatrixXd & XSigmaPoints)
{
   //create sigma point matrix
   XSigmaPoints.setZero(m_n_x, 2 * m_n_x + 1);

   //calculate square root of P
   Eigen::MatrixXd A = P.llt().matrixL();

   //set first column of sigma point matrix
   XSigmaPoints.col(0)  = x;

   //set remaining sigma points
   for (int i = 0; i < m_n_x; i++)
   {
      XSigmaPoints.col(i+1)       = x + sqrt(m_lambda+m_n_x) * A.col(i);
      XSigmaPoints.col(i+1+m_n_x) = x - sqrt(m_lambda+m_n_x) * A.col(i);
   }
}

void Ukf::generateAugumentedSigmaPoints(const Eigen::VectorXd &x, const Eigen::MatrixXd & P, const Eigen::VectorXd &std, Eigen::MatrixXd & XSigmaPoints)
{
   //create augmented mean vector
   Eigen::VectorXd x_aug;
   x_aug.setZero(m_n_x_aug);
   x_aug.head(m_n_x) = x;

   //create augmented state covariance
   Eigen::MatrixXd P_aug;
   P_aug.setZero(m_n_x_aug, m_n_x_aug);
   P_aug.topLeftCorner(m_n_x, m_n_x) = P;

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



const Eigen::ArrayXd Ukf::calculateWeights()
{
   Eigen::ArrayXd weights(m_nSigmaPoints);

   weights(0) = m_lambda / (m_lambda + m_n_x_aug);

   for (int i = 1; i < m_nSigmaPoints; ++i)
   {
      weights(i) = 0.5 / (m_lambda + m_n_x_aug);
   }

   return  weights;
}

void Ukf::weightedMean(const Eigen::MatrixXd & XSigmaPoints, Eigen::VectorXd &x_pred)
{
   x_pred = (XSigmaPoints.array().rowwise() * m_weights.transpose()).rowwise().sum();
}

void Ukf::calculateCrosscorrelationMatix(const Eigen::MatrixXd & XSigmaPointsPred, const Eigen::MatrixXd & ZSigmaPointsPred, const Eigen::VectorXd & x_mean, const Eigen::VectorXd & z_mean, Eigen::MatrixXd & T)
{
   const Eigen::MatrixXd X = (XSigmaPointsPred.colwise() - x_mean);
   const Eigen::MatrixXd Z = (ZSigmaPointsPred.colwise() - z_mean);
   const Eigen::MatrixXd X_weighted = X.array().rowwise() * m_weights.transpose();
   T = (X_weighted * Z.transpose());
}

void Ukf::calculateKalmanGain(const Eigen::MatrixXd & T, const Eigen::MatrixXd & S, Eigen::MatrixXd & K)
{
   K = T * S.inverse();
}

void Ukf::updateState(const Eigen::MatrixXd & K, const Eigen::VectorXd & z, const Eigen::VectorXd & z_pred, Eigen::VectorXd & x_pred)
{
   Eigen::VectorXd & x_update = x_pred;
   x_update = x_pred + K * (z - z_pred);
}

void Ukf::updateCovariance(const Eigen::MatrixXd & K, const Eigen::MatrixXd & S, Eigen::MatrixXd & P_pred)
{
   Eigen::MatrixXd & P_update = P_pred;
   P_update = P_pred - K * S * K.transpose();
}


