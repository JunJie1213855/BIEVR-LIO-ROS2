#ifndef BIEVR_LIO_SPHERE_PARAMETERIZATION_H_
#define BIEVR_LIO_SPHERE_PARAMETERIZATION_H_

#include <ceres/ceres.h>

#include <cmath>

#include "bievr_lio/common.h"

namespace bievr {

// Unit-norm 3-vector living on the 2-sphere, e.g. the gravity direction.
//
// Backport of ceres::SphereManifold<3> for Ceres 2.0: the Manifold API (along with
// Problem::SetManifold) only exists from 2.1 onwards, while Ubuntu 22.04 ships 2.0.
// The semantics are the same -- the step has two degrees of freedom in the tangent
// plane at x, and the result is projected back onto the sphere:
//
//   x' = normalize(x + B(x) * delta),  B(x) = 3x2 orthonormal basis of T_x S^2
//
// Mapping it onto the old API is a rename: LocalParameterization <-> Manifold and
// SetParameterization <-> SetManifold.
class SphereParameterization : public ceres::LocalParameterization {
 public:
  bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
    const V3 x_vec = Eigen::Map<const V3>(x);
    // Copy rather than map: ceres only guarantees the 8-byte alignment of double.
    const Eigen::Vector2d delta_vec(delta[0], delta[1]);

    const V3 y = x_vec + tangentBasis(x_vec) * delta_vec;
    const double norm = y.norm();
    Eigen::Map<V3> x_plus_delta_vec(x_plus_delta);
    // norm == 0 only if x itself were zero, which the sphere excludes.
    x_plus_delta_vec = (norm > 0.0) ? (y / norm) : x_vec;
    return true;
  }

  // The jacobian of Plus(x, delta) with respect to delta, evaluated at delta = 0.
  // There the normalization is stationary, so it is just the tangent basis.
  // Ceres expects a row-major GlobalSize() x LocalSize() matrix.
  bool ComputeJacobian(const double* x, double* jacobian) const override {
    const V3 x_vec = Eigen::Map<const V3>(x);
    Eigen::Map<Eigen::Matrix<double, 3, 2, Eigen::RowMajor>, Eigen::Unaligned> jac(jacobian);
    jac = tangentBasis(x_vec);
    return true;
  }

  int GlobalSize() const override { return 3; }
  int LocalSize() const override { return 2; }

 private:
  // Orthonormal basis of the tangent plane at x, as the columns of a 3x2 matrix.
  // Any basis spans the same tangent plane, so one built by Gram-Schmidt from a
  // coordinate axis is enough; the axis is picked to stay away from the degenerate
  // case of being (anti)parallel to x.
  static Eigen::Matrix<double, 3, 2> tangentBasis(const V3& x) {
    const V3 axis = (std::abs(x.x()) < 0.9) ? V3::UnitX() : V3::UnitY();
    Eigen::Matrix<double, 3, 2> basis;
    basis.col(0) = (axis - x * axis.dot(x)).normalized();
    basis.col(1) = x.cross(basis.col(0)).normalized();
    return basis;
  }
};

}  // namespace bievr

#endif  // BIEVR_LIO_SPHERE_PARAMETERIZATION_H_
