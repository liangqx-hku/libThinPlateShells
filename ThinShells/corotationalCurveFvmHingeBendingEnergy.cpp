#include "ElasticEnergy.h"
#include "../Common/CommonFunctions.h"
#include <tbb/tbb.h>
#include <iostream>

double eleCorotationalCurveFvmHingeBendingEnergy(
    const MeshConnectivity &mesh,
    const Eigen::MatrixXd& iniPos,
    const Eigen::MatrixXd &curPos,
    double YoungsModulus, double PoissonsRatio, double thickness,
    int face, 
    const Eigen::Matrix2d &abar,
    const SecondFundamentalFormDiscretization& sff,
    Eigen::MatrixXd *derivative, 
    Eigen::MatrixXd *hessian,
    bool isLocalProj)
{
    const double planeStressCoef = YoungsModulus / (1 - std::pow(PoissonsRatio, 2));
    Eigen::Matrix<double, 3, 3> constitutiveMat; 
    constitutiveMat << 1, PoissonsRatio, 0,
                        PoissonsRatio, 1, 0,
                        0, 0, 0.5*(1-PoissonsRatio);
    Eigen::Matrix<double, 3, 3> bendingRigidityMat = std::pow(thickness, 3) / 12.0 * planeStressCoef * constitutiveMat;
    
    Eigen::Matrix<int, 1, 6> elemVInd = Eigen::Matrix<int, 1, 6>::Constant(-1);
    for (int i = 0; i < 3; i++)
    {
        elemVInd(i) = mesh.faceVertex(face, i);
        elemVInd(i + 3) = mesh.vertexOppositeFaceEdge(face, i);
    }
    double triArea = 0.5 * sqrt(abar.determinant());
    Eigen::Vector3d X1 = iniPos.row(elemVInd(0)).transpose();
    Eigen::Vector3d X2 = iniPos.row(elemVInd(1)).transpose();
    Eigen::Vector3d X3 = iniPos.row(elemVInd(2)).transpose();
    Eigen::Vector3d X4 = elemVInd(3) != -1 ? iniPos.row(elemVInd(3)).transpose().eval() : (X2 + X3 - X1);
    Eigen::Vector3d X5 = elemVInd(4) != -1 ? iniPos.row(elemVInd(4)).transpose().eval() : (X3 + X1 - X2);
    Eigen::Vector3d X6 = elemVInd(5) != -1 ? iniPos.row(elemVInd(5)).transpose().eval() : (X1 + X2 - X3);
    Eigen::Matrix<double, 3, 6> elePatchRestPos = Eigen::Matrix<double, 3, 6>::Zero();
    elePatchRestPos << X1, X2, X3, X4, X5, X6;
    Eigen::Matrix<double, 3, 1> normal0 = ((X2 - X1).cross(X3 - X1)).normalized();
    Eigen::Matrix<double, 3, 1> local_x_axis = (X3 - X1).normalized();
    Eigen::Matrix<double, 3, 1> local_y_axis = normal0.cross(local_x_axis);
    Eigen::Matrix<double, 3, 3> local_exyzT;
    local_exyzT << local_x_axis.transpose(),
                    local_y_axis.transpose(),
                    normal0.transpose();
    Eigen::Matrix<double, 3, 6> matX1;
    matX1 << X1, X1, X1, X1, X1, X1;
    const auto localXYZ = local_exyzT * (elePatchRestPos - matX1);
    Eigen::Matrix<double, 18, 1> X_e;
    X_e << elePatchRestPos.col(0),
            elePatchRestPos.col(1),
            elePatchRestPos.col(2),
            elePatchRestPos.col(3),
            elePatchRestPos.col(4),
            elePatchRestPos.col(5);
    Eigen::Matrix<double, 6, 18> N0 = Eigen::Matrix<double, 6, 18>::Zero();
    N0.block<1, 3>(0, 0)  = normal0.transpose();
    N0.block<1, 3>(1, 3)  = normal0.transpose();
    N0.block<1, 3>(2, 6)  = normal0.transpose();
    N0.block<1, 3>(3, 9)  = normal0.transpose();
    N0.block<1, 3>(4, 12) = normal0.transpose();
    N0.block<1, 3>(5, 15) = normal0.transpose();
    // corotational curve-fvm-hinge
    Eigen::Matrix<double, 3, 3> TransMat = Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::Matrix<double, 3, 6> curviGradOp = Eigen::Matrix<double, 3, 6>::Zero();
    bool boundaryElem = false;
    Eigen::Matrix<double, 3, 3> newBendingRigidityMat = bendingRigidityMat;
    curviGradOpCurveFvmHinge(mesh, localXYZ, newBendingRigidityMat, boundaryElem, face, TransMat, curviGradOp);
    const Eigen::Matrix<double, 3, 3> TransBendingRigidity = TransMat.transpose() * newBendingRigidityMat * TransMat;
    Eigen::Matrix<double, 3, 1> curvature0 = curviGradOp * N0 * X_e;

    Eigen::Vector3d x1 = curPos.row(elemVInd(0)).transpose();
    Eigen::Vector3d x2 = curPos.row(elemVInd(1)).transpose();
    Eigen::Vector3d x3 = curPos.row(elemVInd(2)).transpose();
    // generate the virtual nodes for free edges
    Eigen::Vector3d x4 = elemVInd(3) != -1 ? curPos.row(elemVInd(3)).transpose().eval() : (x2 + x3 - x1);
    Eigen::Vector3d x5 = elemVInd(4) != -1 ? curPos.row(elemVInd(4)).transpose().eval() : (x3 + x1 - x2);
    Eigen::Vector3d x6 = elemVInd(5) != -1 ? curPos.row(elemVInd(5)).transpose().eval() : (x1 + x2 - x3);
    Eigen::Matrix<double, 3, 1> normal = ((x2 - x1).cross(x3 - x1)).normalized();
    Eigen::Matrix<double, 3, 6> elePatchCurPos = Eigen::Matrix<double, 3, 6>::Zero();
    elePatchCurPos << x1, x2, x3, x4, x5, x6;
    Eigen::Matrix<double, 18, 1> x_e; 
    x_e << elePatchCurPos.col(0),
           elePatchCurPos.col(1),
           elePatchCurPos.col(2),
           elePatchCurPos.col(3),
           elePatchCurPos.col(4),
           elePatchCurPos.col(5);
    Eigen::Matrix<double, 6, 18> N = Eigen::Matrix<double, 6, 18>::Zero();
    N.block<1, 3>(0, 0)  = normal.transpose();
    N.block<1, 3>(1, 3)  = normal.transpose();
    N.block<1, 3>(2, 6)  = normal.transpose();
    N.block<1, 3>(3, 9)  = normal.transpose();
    N.block<1, 3>(4, 12) = normal.transpose();
    N.block<1, 3>(5, 15) = normal.transpose();

    // curvature change
    Eigen::Matrix<double, 3, 1> curvature = curviGradOp * N * x_e;
    Eigen::Matrix<double, 3, 1> curvatureChange = curvature - curvature0;
    Eigen::Matrix<double, 3, 1> moment = TransBendingRigidity * curvatureChange;

    // -------------------  bending energy  -----------------------------------
    double bendingEnergy = 0.5 * triArea * curvatureChange.dot(moment);

    // -------------------  gradient of bending energy  -----------------------
    if (derivative)
    {
        derivative->setZero();
        Eigen::Matrix<double, 3, 18> grad_n = Eigen::Matrix<double, 3, 18>::Zero();
        compute_normal_gradient(X1, X2, X3, x1 - X1, x2 - X2, x3 - X3, grad_n);
        Eigen::Matrix<double, 6, 18> pNpxTx_e;
        pNpxTx_e << x1.transpose() * grad_n,
                    x2.transpose() * grad_n,
                    x3.transpose() * grad_n,
                    x4.transpose() * grad_n,
                    x5.transpose() * grad_n,
                    x6.transpose() * grad_n;
        
        Eigen::Matrix<double, 6, 18> N = Eigen::Matrix<double, 6, 18>::Zero();
        N.block<1, 3>(0, 0) = normal.transpose();
        N.block<1, 3>(1, 3) = normal.transpose();
        N.block<1, 3>(2, 6) = normal.transpose();
        N.block<1, 3>(3, 9) = normal.transpose();
        N.block<1, 3>(4, 12) = normal.transpose();
        N.block<1, 3>(5, 15) = normal.transpose();
        
        *derivative = triArea * moment.transpose() * (curviGradOp * pNpxTx_e + curviGradOp * N);
    }
    
    
    // -----------------  hessian of bending energy  -------------------------
    if (hessian)
    {
        hessian->setZero();
        Eigen::MatrixXd Hess_flatten; Hess_flatten.setZero(18, 18);
        KeroneckerProduct(curviGradOp.transpose()*TransBendingRigidity*curviGradOp, Eigen::Matrix<double, 3, 3>::Identity(), Hess_flatten);
        *hessian = triArea * Hess_flatten;
    }
    
    return bendingEnergy;
}

double corotationalCurveFvmHingeBendingEnergy(
    const MeshConnectivity& mesh,
    const Eigen::MatrixXd& iniPos,
    const Eigen::MatrixXd& curPos,
    double YoungsModulus, double PoissonsRatio, double thickness,
    const std::vector<Eigen::Matrix2d>& abars,
    const SecondFundamentalFormDiscretization& sff,
    Eigen::VectorXd* derivative, // positions, then thetas
    std::vector<Eigen::Triplet<double> >* hessian,
    bool isLocalProj,
    bool isParallel)
{
    int nfaces = mesh.nFaces();
    int nedges = mesh.nEdges();
    int nverts = curPos.rows();
    int nedgedofs = sff.numExtraDOFs();

    if (derivative)
    {
        derivative->resize(3 * nverts + nedgedofs * nedges);
        derivative->setZero();
    }
    if (hessian)
    {
        hessian->clear();
    }

    Eigen::MatrixXi F = mesh.faces();

    double result = 0;
    auto energies = std::vector<double>(nfaces);
    auto derivs = std::vector<Eigen::MatrixXd>(nfaces);
    auto hesses = std::vector<Eigen::MatrixXd>(nfaces);

    if (isParallel)
    {
        auto computeBending = [&](const tbb::blocked_range<uint32_t>& range)
        {
            for (uint32_t i = range.begin(); i < range.end(); ++i)
            {
                Eigen::MatrixXd deriv(1, 18 + 3 * nedgedofs);
                Eigen::MatrixXd hess(18 + 3 * nedgedofs, 18 + 3 * nedgedofs);
                energies[i] = eleCorotationalCurveFvmHingeBendingEnergy(mesh, iniPos, curPos, YoungsModulus, PoissonsRatio, thickness, i, abars[i], sff, derivative ? &deriv : NULL, hessian ? &hess : NULL, isLocalProj);
                if (derivative)
                    derivs[i] = deriv;
                if (hessian)
                    hesses[i] = hess;
            }
        };

        tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nfaces);
        tbb::parallel_for(rangex, computeBending);
    }
    else // serial version
    {
        for (int i = 0; i < nfaces; i++) // 
        {
            Eigen::MatrixXd deriv(1, 18 + 3 * nedgedofs);
            Eigen::MatrixXd hess(18 + 3 * nedgedofs, 18 + 3 * nedgedofs);
            energies[i] = eleCorotationalCurveFvmHingeBendingEnergy(mesh, iniPos, curPos, YoungsModulus, PoissonsRatio, thickness, i, abars[i], sff, derivative ? &deriv : NULL, hessian ? &hess : NULL, isLocalProj);
            if (derivative)
                derivs[i] = deriv;
            if (hessian)
                hesses[i] = hess;
        }
    }
    
    for (int i = 0; i < nfaces; i++)
    {
        result += energies[i];
        if (derivative)
        {
            for (int j = 0; j < 3; j++)
            {
                derivative->segment<3>(3 * mesh.faceVertex(i, j)).transpose() += derivs[i].block<1, 3>(0, 3 * j);
                int oppidx = mesh.vertexOppositeFaceEdge(i, j);
                if (oppidx != -1)
                    derivative->segment<3>(3 * oppidx).transpose() += derivs[i].block<1, 3>(0, 9 + 3 * j);
                for (int k = 0; k < nedgedofs; k++)
                {
                    (*derivative)[3 * nverts + nedgedofs * mesh.faceEdge(i, j) + k] += derivs[i](0, 18 + nedgedofs * j + k);
                }
            }
        }
        if (hessian)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        for (int m = 0; m < 3; m++)
                        {
                            hessian->push_back(Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l, 3 * mesh.faceVertex(i, k) + m, hesses[i](3 * j + l, 3 * k + m)));
                            int oppidxk = mesh.vertexOppositeFaceEdge(i, k);
                            if (oppidxk != -1)
                                hessian->push_back(Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l, 3 * oppidxk + m, hesses[i](3 * j + l, 9 + 3 * k + m)));
                            int oppidxj = mesh.vertexOppositeFaceEdge(i, j);
                            if (oppidxj != -1)
                                hessian->push_back(Eigen::Triplet<double>(3 * oppidxj + l, 3 * mesh.faceVertex(i, k) + m, hesses[i](9 + 3 * j + l, 3 * k + m)));
                            if (oppidxj != -1 && oppidxk != -1)
                                hessian->push_back(Eigen::Triplet<double>(3 * oppidxj + l, 3 * oppidxk + m, hesses[i](9 + 3 * j + l, 9 + 3 * k + m)));
                        }
                        for (int m = 0; m < nedgedofs; m++)
                        {
                            hessian->push_back(Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l, 3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m, hesses[i](3 * j + l, 18 + nedgedofs * k + m)));
                            hessian->push_back(Eigen::Triplet<double>(3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m, 3 * mesh.faceVertex(i, j) + l, hesses[i](18 + nedgedofs * k + m, 3 * j + l)));
                            int oppidxj = mesh.vertexOppositeFaceEdge(i, j);
                            if (oppidxj != -1)
                            {
                                hessian->push_back(Eigen::Triplet<double>(3 * oppidxj + l, 3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m, hesses[i](9 + 3 * j + l, 18 + nedgedofs * k + m)));
                                hessian->push_back(Eigen::Triplet<double>(3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m, 3 * oppidxj + l, hesses[i](18 + nedgedofs * k + m, 9 + 3 * j + l)));
                            }
                        }
                    }
                    for (int m = 0; m < nedgedofs; m++)
                    {
                        for (int n = 0; n < nedgedofs; n++)
                        {
                            hessian->push_back(Eigen::Triplet<double>(3 * nverts + nedgedofs * mesh.faceEdge(i, j) + m, 3 * nverts + nedgedofs * mesh.faceEdge(i, k) + n, hesses[i](18 + nedgedofs * j + m, 18 + nedgedofs * k + n)));
                        }
                    }
                }
            }
        }
    }

    return result;
}


