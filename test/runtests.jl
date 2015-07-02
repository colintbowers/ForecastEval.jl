using ForecastEval
using KernelStat
using ARIMA
using DependentBootstrap
using LossFunctions

const numObs = 500::Int
const numModel = 6::Int
const dmMethodList = [DMAsymptoticBasic(numObs), DMAsymptoticHAC(numObs), DMBootstrap(numObs)]::Vector{DMMethod}
const rcMethodList = [RCBootstrap(numObs), RCBootstrapAlt(numObs)]::Vector{RCMethod}
const spaMethodList = [SPABootstrap(numObs), SPABootstrap(numObs, :epanechnikov)]::Vector{SPABootstrap}

function simlD(maOrder::Int, elD::Vector{Float64})
	maOrder == 0 ? maCoef = Array(Float64, 0) : maCoef = [ 1 / 2^p for p = 1:maOrder ]
	lD = Array(Float64, numObs, length(elD))
	for k = 1:length(elD)
		lD[:, k] = simulate(numObs, ARIMAModel(maCoef=maCoef, intercept=elD[k]))
	end
	ranking = flipud(sortperm(elD))
	return(lD, ranking)
end




function testspa(elD::Vector{Float64}=zeros(Float64, numModel))
	numIter = 100
	for maOrder = 0:0
	#for maOrder = 0:3:6
		for j = 1:1
			spaM = spaMethodList[j]
			println("Method = " * string(spaM))
			println("True MA order = " * string(maOrder))
			println("True Expected LD = " * string(elD))
			pValMat = Array(Float64, 3, numIter)
			for n = 1:numIter
				(lD, ranking) = simlD(maOrder, elD)
				pValMat[:, n] = spa(lD, method=spaM)
			end
			println("spa_u rej freq = " * string((1/numIter) * sum(vec(pValMat[1, :]) .< 0.05)))
			println("spa_l rej freq = " * string((1/numIter) * sum(vec(pValMat[2, :]) .< 0.05)))
			println("spa_c rej freq = " * string((1/numIter) * sum(vec(pValMat[3, :]) .< 0.05)))
			println("")
		end
	end
	return(true)
end



#H0: The basecase is not worse than any of the models
#H0: The basecase is at least as good, if not better, than all of the models
#HA: At least one model is better than the basecase
#H0: No model better than base-case

#SPA test in current form will not work when all models are significantly worse than the base case. At least one model must be as good or better. Not sure why this isn't mentioned in the paper

testspa()
testspa(0.5 * ones(Float64, numModel)) #H0 is true (rej freq -> 0.0)
testspa(-0.5 * ones(Float64, numModel)) #HA is true (rej freq -> 1.0)

testspa(0.1 * ones(Float64, numModel)) #base-case better
testspa(-0.1 * ones(Float64, numModel)) #models better

testspa([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
testspa(-0.05 * ones(Float64, numModel))

testspa(0.11 * ones(Float64, numModel))
testspa([0.01, 0.5 * ones(Float64, numModel-1)])


showcompact(randn())

function testspaOld(baseCaseMult::Float64=1.0)
	for maOrder = 6:6
	#for maOrder = 0:3:6
		(xTrue, vbc, bc, vModel, x, ranking) = simmodel(maOrder, 0.1, 0.25, baseCaseMult)
		x = [ x[m][n] for n = 1:numObs, m = 1:numModel ]
		#for j = 1:length(spaMethodList)
		for j = 1:1
			spaM = spaMethodList[j]
			pValVec = spa(x, bc, xTrue, method=spaM)
			println("Method = " * string(spaM))
			println("True MA order = " * string(maOrder))
			println("True ranking = " * string(ranking))
			println("Base-case error variance = " * string(vbc))
			println("Model error variances = " * string(vModel))
			println("spa pval_u = " * string(pValVec[1]))
			println("spa pval_l = " * string(pValVec[2]))
			println("spa pval_c = " * string(pValVec[3]))
			println("")
			println("")
		end
	end
	return(true)
end






function testrc()
#	for maOrder = 0:3:6
	for maOrder = 6:6
		(xTrue, vbc, bc, vModel, x, ranking) = simmodel(maOrder)
		x = col2mat(x)
		for j = 1:length(rcMethodList)
			rcM = rcMethodList[j]
			pValVec = rc(x, bc, xTrue, method=rcM)
			println("Method = " * string(rcM))
			println("True MA order = " * string(maOrder))
			println("True ranking = " * string(ranking))
			println("Base-case error variance = " * string(vbc))
			println("Model error variances = " * string(vModel))
			println("rc p-values = " * string(pValVec))
			println("")
			println("")
		end
	end
	return(true)
end






function simmodel(maOrder::Int, eVarMin::Float64=0.1, eVarMax::Float64=0.25, baseCaseMult::Float64=1.0)
	maOrder == 0 ? maCoef = Array(Float64, 0) : maCoef = [ 1 / 2^p for p = 1:maOrder ]
	xTrue = simulate(numObs, ARIMAModel(maCoef=maCoef))
	vbc = eVarMin + (eVarMax - eVarMin) * rand()
	vbc = baseCaseMult * vbc
	vModel = eVarMin + (eVarMax - eVarMin) * rand(numModel)
	bc = xTrue + vbc * randn(numObs)
	x = [ xTrue + vModel[n] * randn(numObs) for n = 1:numModel ]
	ranking = sortperm([vModel, vbc])
	ranking[ranking .== numModel+1] = 0
	return(xTrue, vbc, bc, vModel, x, ranking)
end
col2mat{T}(x::Vector{Vector{T}}) = [ x[m][n] for n = 1:length(x[1]), m = 1:length(x) ]
mat2col{T}(x::Matrix{T}) = [ x[:, m] for m = 1:size(x, 2) ]



function testdm(elD::Float64=0.0)
	numIter = 100
	if elD < 0.0; trueRanking = 1
	elseif elD == 0.0; trueRanking = 0
	else; trueRanking = -1
	end
#	for maOrder = 0:1:20
	for maOrder = 0:3:6
		for j = 2:length(dmMethodList)
			dmM = dmMethodList[j]
			println("dm Method = " * string(dmM))
			println("True MA order = " * string(maOrder))
			println("True ranking = " * string(trueRanking))
			pValVec = Array(Float64, numIter)
			dmRankVec = Array(Int, numIter)
			for n = 1:numIter
				(lD, ranking) = simlD(maOrder, [elD])
				lD = vec(lD)
				(pValVec[n], dmRankVec[n]) = dm(lD, method=dmM)
			end
			println("    Fail to reject proportion = " * string(sum(dmRankVec .== 0) / numIter))
			println("    Reject in favour of model 1 = " * string(sum(dmRankVec .== 1) / numIter))
			println("    Reject in favour of model 2 = " * string(sum(dmRankVec .== -1) / numIter))
# 			println("    p-value list:")
# 			for n = 1:numIter
# 				println("        " * string(pValVec[n]))
# 			end
		end
	end
	return(true)
end








































