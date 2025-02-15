#include "ElasticEnergy.h"
#include "../Common/CommonFunctions.h"
#include <tbb/tbb.h>
#include <iostream>

void curviGradOpFlatFvmHinge(const MeshConnectivity &mesh, const Eigen::Matrix<double, 3, 6>& localXYZ, 
                              Eigen::Matrix<double, 3, 3>& bendingRigidityMat, bool& boundaryElem, int face,
                              Eigen::Matrix<double, 3, 3>& TransMat, Eigen::Matrix<double, 3, 6>& curviGradOp) 
{
    Eigen::Matrix<int, 3, 4> hingeVInds;
    hingeVInds << 2, 1, 3, 0,
                  0, 2, 4, 1,
                  1, 0, 5, 2;
    for (int edgeID = 0; edgeID < 3; ++edgeID)
    {
        Eigen::RowVector4i hingeVInd = hingeVInds.row(edgeID);
        Eigen::Matrix<double, 3, 4> hingeRestPos;
        hingeRestPos << localXYZ.col(hingeVInd(0)), localXYZ.col(hingeVInd(1)), localXYZ.col(hingeVInd(2)), localXYZ.col(hingeVInd(3));
        Eigen::Matrix<double, 3, 1> XA = hingeRestPos.col(0), XB = hingeRestPos.col(1), XC = hingeRestPos.col(2), XD = hingeRestPos.col(3);
        Eigen::Matrix<double, 3, 1> iniEdgeAB = XB - XA, iniEdgeAC = XC - XA, iniEdgeAD = XD - XA;
        const double restLenAB = iniEdgeAB.norm();
        const Eigen::Matrix<double, 3, 1> dirRestAB = iniEdgeAB / restLenAB; 
        const double PA = iniEdgeAC.dot(iniEdgeAB) / restLenAB;
        const double PB = restLenAB - PA;
        const double PC = (PA*dirRestAB - iniEdgeAC).norm(); 
        const double QA = iniEdgeAD.dot(iniEdgeAB) / restLenAB;
        const double QB = restLenAB - QA;
        const double QD = (QA*dirRestAB - iniEdgeAD).norm(); 
        Eigen::Matrix<double, 3, 1> m = (QA*dirRestAB - iniEdgeAD) / QD;
        const double cosPhi = m.x();
        const double sinPhi = m.y();
        TransMat.col(edgeID) << pow(cosPhi, 2), 
                                pow(sinPhi, 2), 
                                2 * cosPhi*sinPhi;
        Eigen::Matrix<double, 1, 6> L = Eigen::Matrix<double, 1, 6>::Zero();
        const double factor = 2 / (PC + QD);
        const double LA = -(PB/PC + QB/QD)/restLenAB;
        const double LB = -(PA/PC + QA/QD)/restLenAB;
        const double LC =  1.0/PC;
        const double LD =  1.0/QD;
        L(hingeVInd(0)) = factor*LA; L(hingeVInd(1)) = factor*LB; L(hingeVInd(2)) = factor*LC; L(hingeVInd(3)) = factor*LD;
        curviGradOp.row(edgeID) = L;

        // boundary condition
        // free boundary edge
        int oppVID = mesh.vertexOppositeFaceEdge(face, edgeID);
        if (oppVID == -1) // if this is a boundary edge, then
        {
            boundaryElem = true;
            if(true)
            {
                // if this is a free boundary edge, then
                curviGradOp.row(edgeID).setZero();
                bendingRigidityMat(2, 2) = 0.0;
            }
            
            if(false)
            {
                // if this is a clamp boundary edge, then
                const double factor = 2 / (PC + QD);
                const double LA = -2*QB/QD/restLenAB;
                const double LB = -2*QA/QD/restLenAB;
                const double LC =  0.0;
                const double LD =  2.0/QD;
                L(hingeVInd(0)) = factor*LA; L(hingeVInd(1)) = factor*LB; L(hingeVInd(2)) = factor*LC; L(hingeVInd(3)) = factor*LD;
                curviGradOp.row(edgeID) = L;
            }
        }
    }
}


double corotationalFlatFvmHingeBendingEnergy(
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
    int dofs_ = 3 * nverts + nedgedofs * nedges;
    Eigen::VectorXd disp(3 * nverts); // displacement
    for (int i = 0; i < nverts; i++)
        disp.segment<3>(3 * i) = curPos.row(i) - iniPos.row(i);

    if (derivative)
    {
        derivative->resize(dofs_);
        derivative->setZero();
    }
    if (hessian)
    {
        hessian->clear();
    }

    std::vector<Eigen::Triplet<double>> bendingStiffK;
    bendingStiffK.clear();
    bendingStiffK.resize(nfaces * 18 * 18); 
    const double planeStressCoef = YoungsModulus / (1 - std::pow(PoissonsRatio, 2));
    Eigen::Matrix<double, 3, 3> constitutiveMat; 
    constitutiveMat << 1, PoissonsRatio, 0,
                        PoissonsRatio, 1, 0,
                        0, 0, 0.5*(1-PoissonsRatio);
    Eigen::Matrix<double, 3, 3> bendingRigidityMat = std::pow(thickness, 3) / 12.0 * planeStressCoef * constitutiveMat;
    if (isParallel)
    {
        auto computeBending = [&](const tbb::blocked_range<uint32_t>& range)
        {
            for (uint32_t face = range.begin(); face < range.end(); ++face)
            {
                Eigen::Matrix<int, 1, 6> elemVInd = Eigen::Matrix<int, 1, 6>::Constant(-1);
                for (int i = 0; i < 3; i++)
                {
                    elemVInd(i) = mesh.faceVertex(face, i);
                    elemVInd(i + 3) = mesh.vertexOppositeFaceEdge(face, i);
                }
                double triArea = 0.5 * sqrt(abars[face].determinant());
                Eigen::Vector3d X1 = iniPos.row(elemVInd(0)).transpose();
                Eigen::Vector3d X2 = iniPos.row(elemVInd(1)).transpose();
                Eigen::Vector3d X3 = iniPos.row(elemVInd(2)).transpose();
                Eigen::Vector3d X4 = elemVInd(3) != -1 ? iniPos.row(elemVInd(3)).transpose().eval() : (X2 + X3 - X1);
                Eigen::Vector3d X5 = elemVInd(4) != -1 ? iniPos.row(elemVInd(4)).transpose().eval() : (X3 + X1 - X2);
                Eigen::Vector3d X6 = elemVInd(5) != -1 ? iniPos.row(elemVInd(5)).transpose().eval() : (X1 + X2 - X3);
                Eigen::Matrix<double, 3, 6> elePatchRestPos = Eigen::Matrix<double, 3, 6>::Zero();
                elePatchRestPos << X1, X2, X3, X4, X5, X6;
                Eigen::Matrix<double, 3, 6> localXYZ = corotationalFrame(X1, X2, X3, elePatchRestPos);
                Eigen::Matrix<double, 3, 3> TransMat = Eigen::Matrix<double, 3, 3>::Zero();
                Eigen::Matrix<double, 3, 6> curviGradOp = Eigen::Matrix<double, 3, 6>::Zero();
                bool boundaryElem = false;
                Eigen::Matrix<double, 3, 3> newBendingRigidityMat = bendingRigidityMat;
                curviGradOpFlatFvmHinge(mesh, localXYZ, newBendingRigidityMat, boundaryElem, face, TransMat, curviGradOp);

                const Eigen::Matrix<double, 3, 3> TransBendingRigidity = TransMat.transpose() * newBendingRigidityMat * TransMat;
                Eigen::MatrixXd TransBendingRigidity_flatten;
                TransBendingRigidity_flatten.resize(9, 9); TransBendingRigidity_flatten.setZero();
                KeroneckerProduct(TransBendingRigidity, Eigen::Matrix<double, 3, 3>::Identity(3, 3), TransBendingRigidity_flatten); 
                Eigen::MatrixXd pAssmCurpUT;
                pAssmCurpUT.resize(9, 18); pAssmCurpUT.setZero();
                KeroneckerProduct(curviGradOp, Eigen::Matrix<double, 3, 3>::Identity(), pAssmCurpUT);

                Eigen::Matrix<double, 18, 18> elementalBendingStiffMatrix = triArea * pAssmCurpUT.transpose() * TransBendingRigidity_flatten * pAssmCurpUT;
                // if (isLocalProj) 
                //     elementalBendingStiffMatrix = lowRankApprox(elementalBendingStiffMatrix);
                const int offset = face * 6 * 3 * 6 * 3;
                for (int k = 0; k < 6; ++k)
                    for (int d = 0; d < 3; ++d)
                        for (int s = 0; s < 6; ++s)
                            for (int t = 0; t < 3; ++t){
                                if (elemVInd(k) != -1 && elemVInd(s) != -1) 
                                {
                                    bendingStiffK[offset
                                    + k * 3 * 6 * 3
                                    + d * 6 * 3
                                    + s * 3
                                    + t] = Eigen::Triplet<double>(3 * elemVInd(k) + d, 3 * elemVInd(s) + t, elementalBendingStiffMatrix(s * 3 + t, k * 3 + d));
                                }
                            }

            } 
        };

        tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nfaces);
        tbb::parallel_for(rangex, computeBending);
    }
    else // serial version
    {
        for (int face = 0; face < nfaces; ++face) 
        {
            Eigen::Matrix<int, 1, 6> elemVInd = Eigen::Matrix<int, 1, 6>::Constant(-1);
            for (int i = 0; i < 3; i++)
            {
                elemVInd(i) = mesh.faceVertex(face, i);
                elemVInd(i + 3) = mesh.vertexOppositeFaceEdge(face, i);
            }
            double triArea = 0.5 * sqrt(abars[face].determinant());
            Eigen::Vector3d X1 = iniPos.row(elemVInd(0)).transpose();
            Eigen::Vector3d X2 = iniPos.row(elemVInd(1)).transpose();
            Eigen::Vector3d X3 = iniPos.row(elemVInd(2)).transpose();
            Eigen::Vector3d X4 = elemVInd(3) != -1 ? iniPos.row(elemVInd(3)).transpose().eval() : (X2 + X3 - X1);
            Eigen::Vector3d X5 = elemVInd(4) != -1 ? iniPos.row(elemVInd(4)).transpose().eval() : (X3 + X1 - X2);
            Eigen::Vector3d X6 = elemVInd(5) != -1 ? iniPos.row(elemVInd(5)).transpose().eval() : (X1 + X2 - X3);
            Eigen::Matrix<double, 3, 6> elePatchRestPos = Eigen::Matrix<double, 3, 6>::Zero();
            elePatchRestPos << X1, X2, X3, X4, X5, X6;
            Eigen::Matrix<double, 3, 6> localXYZ = corotationalFrame(X1, X2, X3, elePatchRestPos);
            Eigen::Matrix<double, 3, 3> TransMat = Eigen::Matrix<double, 3, 3>::Zero();
            Eigen::Matrix<double, 3, 6> curviGradOp = Eigen::Matrix<double, 3, 6>::Zero();
            bool boundaryElem = false;
            Eigen::Matrix<double, 3, 3> newBendingRigidityMat = bendingRigidityMat;
            curviGradOpFlatFvmHinge(mesh, localXYZ, newBendingRigidityMat, boundaryElem, face, TransMat, curviGradOp);

            const Eigen::Matrix<double, 3, 3> TransBendingRigidity = TransMat.transpose() * newBendingRigidityMat * TransMat;
            Eigen::MatrixXd TransBendingRigidity_flatten;
            TransBendingRigidity_flatten.resize(9, 9); TransBendingRigidity_flatten.setZero();
            KeroneckerProduct(TransBendingRigidity, Eigen::Matrix<double, 3, 3>::Identity(3, 3), TransBendingRigidity_flatten); 
            Eigen::MatrixXd pAssmCurpUT;
            pAssmCurpUT.resize(9, 18); pAssmCurpUT.setZero();
            KeroneckerProduct(curviGradOp, Eigen::Matrix<double, 3, 3>::Identity(), pAssmCurpUT);

            Eigen::Matrix<double, 18, 18> elementalBendingStiffMatrix = triArea * pAssmCurpUT.transpose() * TransBendingRigidity_flatten * pAssmCurpUT;
            // if (isLocalProj) 
            //     elementalBendingStiffMatrix = lowRankApprox(elementalBendingStiffMatrix);
            const int offset = face * 6 * 3 * 6 * 3;
            for (int k = 0; k < 6; ++k)
                for (int d = 0; d < 3; ++d)
                    for (int s = 0; s < 6; ++s)
                        for (int t = 0; t < 3; ++t){
                            if (elemVInd(k) != -1 && elemVInd(s) != -1) 
                            {
                                bendingStiffK[offset
                                + k * 3 * 6 * 3
                                + d * 6 * 3
                                + s * 3
                                + t] = Eigen::Triplet<double>(3 * elemVInd(k) + d, 3 * elemVInd(s) + t, elementalBendingStiffMatrix(s * 3 + t, k * 3 + d));
                            }
                        }

        }
    }
    
    Eigen::SparseMatrix<double> globalBendSparseStiffMat(3 * nverts, 3 * nverts);
    globalBendSparseStiffMat.setFromTriplets(bendingStiffK.begin(), bendingStiffK.end());

    Eigen::VectorXd gradBendE(3 * nverts);
    gradBendE = globalBendSparseStiffMat * disp;

    const double bendingEnergy = 0.5 * disp.dot(gradBendE);

    if(derivative){
        (*derivative).segment(0, 3 * nverts) = gradBendE;
    }

    if(hessian){
        (*hessian).resize(nfaces * 18 * 18);
        (*hessian) = bendingStiffK;
    }

    return bendingEnergy;
}


