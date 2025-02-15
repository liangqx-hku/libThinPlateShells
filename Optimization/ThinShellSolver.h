#pragma once
#include "../MeshLib/GeometryDerivatives.h"
#include "../ThinShells/ElasticShellModel.h"
#include "../SecondFundamentalForm/SecondFundamentalFormDiscretization.h"

#include "../Common/CommonFunctions.h"


struct FullSimOptimizationParams
{
	size_t iterations = 1000; // maximum number of Newton iterations
	double fDelta = 0; // function update tolerance
	double gradNorm = 1e-3; // gradient tolerance
	double xDelta = 0; // variable update tolerance
	bool isProjH = true; // whether to use positive definite fix
	bool isParallel = true; // whether to use parallel computation
	bool printLog = true; // whether to print log
};

// in libThinShells, the density is the volume density now.
namespace ThinShellSolver
{
	void fullSimNewtonStaticSolver(const ElasticSetup& setup, ElasticState& curState, std::string filePrefix, const FullSimOptimizationParams& params);
	void quasiStaticNewtonSolver(const ElasticSetup& setup, ElasticState& curState, std::string filePrefix, const FullSimOptimizationParams& params);
	void linearPlateBending(const ElasticSetup& setup, ElasticState& curState, std::string filePrefix, const FullSimOptimizationParams& params);
};
