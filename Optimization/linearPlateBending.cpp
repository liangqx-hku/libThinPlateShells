#include  <igl/boundary_loop.h>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include "ThinShellSolver.h"
#include "../Common/Timer.h"

void ThinShellSolver::linearPlateBending(const ElasticSetup& setup, ElasticState& curState, std::string filePrefix, const FullSimOptimizationParams& params)
{
    ElasticShellModel model;
	bool ok = model.initialization(setup, curState, filePrefix, params.isProjH, params.isParallel);
	if (!ok)
    {
        std::cout << "initialization failed." << std::endl;
        return;
    }

    Eigen::VectorXd initX; 
	model.convertCurState2Variables(curState, initX);

    Eigen::VectorXd exterForces = model.externalForces(initX);
    Eigen::SparseMatrix<double> hess = model.bendingHessian(initX);
    Eigen::CholmodSimplicialLLT<Eigen::SparseMatrix<double> > solver(hess);
    Eigen::VectorXd du = solver.solve(exterForces);
    model.convertVariables2CurState(initX+du, curState); 
    const int numNodes = curState.curPos.rows();
    Eigen::MatrixXd dispMat = curState.curPos - curState.initialGuess;
}