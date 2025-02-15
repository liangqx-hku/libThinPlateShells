#pragma once

#include "ElasticSetup.h"
#include "StVKMaterial.h"
#include "NeoHookeanMaterial.h"
#include "StVKTensionFieldMaterial.h"
#include "ElasticShellMaterial.h"

double curveSmoothedHingeBendingEnergy(
    const MeshConnectivity &mesh,
    const Eigen::MatrixXd &iniPos,
    const Eigen::MatrixXd &curPos,
    double lameAlpha, double lameBeta, double thickness,
    const std::vector<Eigen::Matrix2d>& abars,
    const SecondFundamentalFormDiscretization& sff,
    Eigen::VectorXd* derivative, // positions, then thetas
    std::vector<Eigen::Triplet<double> >* hessian,
    bool isLocalProj,
    bool isParallel);

