#ifndef COMMONFUNCTIONS_H
#define COMMONFUNCTIONS_H

#include <set>

#include <iostream>
#include <fstream>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "../SecondFundamentalForm/SecondFundamentalFormDiscretization.h"
#include "../SecondFundamentalForm/MidedgeAngleSinFormulation.h"
#include "../SecondFundamentalForm/MidedgeAngleTanFormulation.h"
#include "../SecondFundamentalForm/MidedgeAverageFormulation.h"

struct QuadraturePoints
{
	double u;
	double v;
	double weight;
};


enum StretchingType
{
	StVK = 0,
	tensionField = 1,
	NeoHookean = 2
};

enum BendingType
{
	mideEdgeBending = 0,
	quadraticBending = 1,
    cubicShell = 2,
    corotationalHinge = 3,
    corotationalCurveHinge = 4,
    corotationalFlatFvmHinge = 5,
    corotationalCurveFvmHinge = 6,
    flatSmoothedHinge = 7,
    curveSmoothedHinge = 8,
	noBending = 9
};

enum SolverType
{
	Lbfgs = 0,
	Newton = 1,
	ActiveSet = 2
};

enum SFFType
{
	MidedgeAverage = 0,
	MidedgeSin = 1,
	MidedgeTan = 2
};

enum SubdivisionType
{
	Midpoint = 0,
	Loop = 1
};

enum RoundingType
{
	ComisoRound = 0,
	GurobiRound = 1
};

struct TimeCost
{
	double gradTime = 0;
	double hessTime = 0;
	double solverTime = 0;
	double collisionDectionTime = 0;
	double lineSearchTime = 0;
	double updateTime = 0;
	double convergenceCheckTime = 0;
	double savingTime = 0;
	double unconstrainedLLTTime = 0;

	double totalTime() 
	{
		return gradTime + hessTime + solverTime + collisionDectionTime + lineSearchTime + updateTime + convergenceCheckTime; // exclude the saving time
	}
};

Eigen::MatrixXd lowRankApprox(Eigen::MatrixXd A);      // semi-positive projection of a symmetric matrix A

Eigen::MatrixXd lowRankApproxBend(Eigen::MatrixXd A, bool& flag);

double cotan(const Eigen::Vector3d v0, const Eigen::Vector3d v1, const Eigen::Vector3d v2);

void locatePotentialPureTensionFaces(const std::vector<Eigen::Matrix2d>& abars, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, std::set<int>& potentialPureTensionFaces);
void getPureTensionVertsEdges(const std::set<int>& potentialPureTensionFaces, const Eigen::MatrixXi& F, std::set<int>* pureTensionEdges, std::set<int> *pureTensionVerts);

Eigen::MatrixXd nullspaceExtraction(const Eigen::SparseMatrix<double> A);

void trivialOffset(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd &offsettedV, double d); // offfset V along its normal direction by a given distance

void rigidBodyAlignment(const Eigen::MatrixXd& tarPos, const MeshConnectivity& mesh, const Eigen::MatrixXd &pos, Eigen::Matrix3d &R, Eigen::Vector3d &t); // implement a really naive iterative closest point (ICP) algorithm for rigid body alignment.

void matToVec(const Eigen::MatrixXd& mat, Eigen::VectorXd& vec);
void vecToMat(const Eigen::VectorXd& vec, Eigen::MatrixXd& mat);

bool mkdir(const std::string& foldername);

// bending utility functions
void curviGradTriStencilOperator(const MeshConnectivity &mesh, const Eigen::Matrix<double, 3, 6>& localXYZ,
                                 Eigen::Matrix<double, 3, 3>& bendingRigidityMat, bool& boundaryElem, int face,
                                 Eigen::Matrix<double, 3, 3>& TransMat, Eigen::Matrix<double, 3, 6>& curviGradOp);

double computeArea(const Eigen::Vector2d& X0, const Eigen::Vector2d& X1, const Eigen::Vector2d& X2);

const Eigen::Matrix<double, 3, 6> corotationalFrame(const Eigen::Matrix<double, 3, 1>& X1, const Eigen::Matrix<double, 3, 1>& X2, const Eigen::Matrix<double, 3, 1>& X3,
                                                    const Eigen::Matrix<double, 3, 6>& elePatchRestPos);

void curviGradOpCurveFvmHinge(const MeshConnectivity &mesh, const Eigen::Matrix<double, 3, 6>& localXYZ, 
                              Eigen::Matrix<double, 3, 3>& bendingRigidityMat, bool& boundaryElem, int face,
                              Eigen::Matrix<double, 3, 3>& TransMat, Eigen::Matrix<double, 3, 6>& curviGradOp);

void compute_normal_gradient(const Eigen::Matrix<double, 3, 1>& node4, const Eigen::Matrix<double, 3, 1>& node5, const Eigen::Matrix<double, 3, 1>& node6,
                             const Eigen::Matrix<double, 3, 1>& nodalU4, const Eigen::Matrix<double, 3, 1>& nodalU5, const Eigen::Matrix<double, 3, 1>& nodalU6,
                             Eigen::Matrix<double, 3, 18>& grad_n);

void KeroneckerProduct(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& C);

const double cotTheta(const Eigen::Matrix<double, 3, 1>& n1, const Eigen::Matrix<double, 3, 1>& n2);

const Eigen::Matrix<double, 3, 3> crossProductMatrix(const Eigen::Matrix<double, 3, 1>& v);

#endif