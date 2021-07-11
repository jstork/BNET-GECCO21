library(reticulate)
library(pbapply)
library(CEGO)
library(keras)

#setwd("~/Google Drive/Work/beNEMOresults")
source("bnet_functions.R")

#use_python("/Library/Frameworks/Python.framework/Versions/3.7/bin/python3", required = TRUE) #local
use_python("/usr/bin/python3.8", required=TRUE) #server
dyn.load("cgp_R.so")
Rgym <- import("gym")

seed <- as.integer(Sys.getenv("PBS_ARRAYID")) #server
if(is.na(seed)) seed <- 12345
set.seed(seed)

##function options, experimental design
bootstrapCritic=FALSE
advantageTD=TRUE
experienceSelectionSize <- 1
trainingFactor <- 1.0

##Logging
diskLogging=TRUE
logID <- floor(runif(1)*1000)
instanceName=paste("test",logID,sep="")

env = Rgym$make('CartPole-v1')
env$reset()

##params
##Fitness function max=1, min=-1
fitSign=1

numInputs <- 4 #70input+1Bias
numOutputs <- 2
numSamples <- 1
funcSet <- c("tanh","sig","gauss","soft","step","relu")
params <- c(numInputs,numNodes=200,numOutputs,nodeArity=10)
#criticParams <- c(numInputs=4,numNodes=200,numOutputs=1,nodeArity=10)
popSize <- 5
initSize <- popSize
elitistSelectionSize <- popSize
elitistArchiveSize <- popSize*2

###Iterative Variables init
fitness <- NULL

##elitists
eliteExperience <- NULL
elitistArchive <- NULL
fitnessElite <- rep(0,elitistSelectionSize)
lenArchive <-0

##model archives
annArchive <- NULL
fitnessArchive<- NULL
inputArchive <- NULL
advantageArchive <- NULL

##experience
allResDiscounted <- NULL

##elitist selection
eliteRepeats <- 0
eliteFitAnn <- NULL
annsFitness <- NULL
###logging
bestMethodCounter <- NULL
bestImprovementMethodCounter <- NULL
totalRepeats <- 0

###INIT of first gen ANNS  (OFFLINE possible, TODO)
offlineExperience <- NULL
if(is.null(offlineExperience)) {
  anns <- lapply(1:initSize, function(x) .Call("initChromoR",x*floor(runif(1)*1000),params,funcSet))
  for (id in 1:initSize) {
    anns[[id]]$id <- id
  }
} else {
  anns <- learnFromOfflineExperience(offlineExperience)
}

currRes <- lapply(anns,computeFitnessStates,steps=200,instances=1,stochasticExploration=TRUE,fitSign,explorationRate = 0.5)

###Function LOOP
for (i in 1:30) {
  print(paste("Iteration:",i))
  ####### Testing and generating EXPERIENCE, use stochastic policy for init gen
  #if(i>1) {
  #currRes <- lapply(anns,computeFitnessStates,steps=200,instances=1,stochasticPolicy=FALSE,fitSign)
  #}
  
  #Testing
  if(i>1) {
    currRes <- lapply(anns[-5],computeFitnessStates,steps=200,instances=1,stochasticPolicy=FALSE)
    exploreRes <- computeFitnessStates(anns[[5]],steps=200,instances=1,stochasticExploration=TRUE,explorationRate = 0.2)
    currRes <- c(currRes,list(exploreRes))
  }
  
  ###### FIND Robust Elitist / do Repeats #### 
  annsFitness <- sapply(currRes, FUN= '[[',1) ## for pendulum (all rewards=1)
  print(annsFitness)
  #annsFitness <- sapply(currRewards, FUN= '[',1)  ### for problems with rewards != 1
  best <- order(annsFitness,decreasing = TRUE)[1:elitistSelectionSize] 
  
  annArchive <- c(annArchive,anns)
  fitnessArchive <- c(fitnessArchive,annsFitness)
  
  if(i>1) {
    ### repeats elitist and challenger ####TODO
    totalRepeats <- totalRepeats+1
    eliteRepeats <- c(eliteRepeats,annsFitness[1])
    ##elitist (first in pop "anns") cannot be a challenger!!
    challenger <- which(annsFitness[-1] > mean(eliteRepeats))+1
    challenger <- challenger[order(annsFitness[challenger],decreasing = TRUE)]
    numChallenger <- length(challenger)
    print(paste("NumberChallengers:",numChallenger,"challengerID:"))
    print(challenger)
    if(numChallenger > 0) {
      for (j in 1:numChallenger) {
        len <- length(eliteRepeats)
        ####
        ##max repeats=3
        len <- min(c(len,3))
        challengerFitness <- annsFitness[challenger[j]]
        if(challengerFitness <= mean(eliteRepeats)) {
          break
        }
        if (len < 3) {
          #eliteRepeats <- c(eliteRepeats,testInstance(eliteFitAnn,instances = 1,stochastic=FALSE))  
          eliteRes <- computeFitnessStates(eliteFitAnn,steps=200,instances=1,stochasticPolicy=FALSE,fitSign)
          eliteRepeats <- c(eliteRepeats,eliteRes$epReward)
          len=3
          ###Update currRes and add to Archive
          currRes <- c(currRes,list(eliteRes))
          anns <- c(anns,list(eliteFitAnn))
          annArchive <- c(annArchive,list(eliteFitAnn))
          fitnessArchive <- c(fitnessArchive,eliteRes$epReward)
        }
        for (repeats in 2:len) {
          #challengerFitness <- c(challengerFitness,testInstance(anns[[best[1]]],instances = 1,stochastic=FALSE))
          challengerRes <- computeFitnessStates(anns[[challenger[j]]],steps=200,instances=1,stochasticPolicy=FALSE,fitSign)
          challengerFitness <- c(challengerFitness,challengerRes$epReward)
          ###Update currRes and add to Archive
          currRes <- c(currRes,list(challengerRes))
          anns <- c(anns,list(anns[[challenger[j]]])) ###TODO better Option??? 
          annArchive <- c(annArchive,list(anns[[challenger[j]]]))
          fitnessArchive <- c(fitnessArchive,challengerRes$epReward)
          if(!(var(eliteRepeats) == 0 & var(challengerFitness)==0)) {
            pValue<- t.test(eliteRepeats,challengerFitness)$p.value
          } else {
            break
          }
          if(pValue < 0.15 & repeats > 2) {
            print(paste("Repeats:",repeats,"pValue:",pValue))
            break
          }
        }
        totalRepeats <- totalRepeats + (repeats-1)
        if (mean(challengerFitness) > mean(eliteRepeats)) {
          print(paste("Selected New Best:",mean(challengerFitness),"OLD:",mean(eliteRepeats)))
          print(eliteRepeats)
          print(challengerFitness)
          eliteRepeats <- challengerFitness
          eliteFitAnn <- anns[[challenger[j]]]
        } else {
          print(paste("Selected Old Best:",mean(eliteRepeats), "NEW:", mean(challengerFitness)))
        }
        ###Update annsFitness in Case of Repeats
        annsFitness <- sapply(currRes, FUN= '[[',1)
        best <- order(annsFitness,decreasing = TRUE)[1:elitistSelectionSize] 
      }
    } else {
      print(paste("Selected Old Best:",mean(eliteRepeats)))
    }
  } else {
    print(paste("Best Init:",annsFitness[best[1]]))
    eliteRepeats <- annsFitness[best[1]]
    eliteFitAnn <- anns[[best[1]]]
  }
  
  
  ############# compute discounted Return (R(t))
  currResDiscounted <- expectedRewardsDiscounted(currRes,discountFactor = 0.99) ###careful here 0.99 is default, used 0.9 before (worked!)
  currInputs <- lapply(currResDiscounted, FUN= '[[',2)
  currRewards <- lapply(currResDiscounted, FUN= '[[',4)
  allResDiscounted <- c(allResDiscounted,currResDiscounted)
  
  ######## LOGGING and Printing
  if(i>1)
    print("Fitness current Gen (first 5) + Repeats (remaining)")
  
  print(annsFitness)
  ##normal plotting
  fitness <- c(fitness,annsFitness)
  plot(fitness,main=(paste("Total Evaluations:",length(fitness),
                           "Total Repeats:",totalRepeats,
                           " Elitist Fitness(avg):",mean(eliteRepeats),
                           " Elitist Repeated:",length(eliteRepeats))),ylim=c(10,200))
  abline(h=200, col="red")
  abline(v=initSize, col="blue")
  ####Stopping Criteria
  
  print("Testing Stopping Criteria")
  testing <- sapply(anns[1:popSize],testInstance,instances=100,fitSign=fitSign)
  print(testing)
  if (i>1) {
    bestMethodCounter <- c(bestMethodCounter,which.max(testing))
    print("Best Method Counter:")
    print(bestMethodCounter)
  }
  if(any(testing >=195 )) {
    print("Found Solution")
    break
  }
  
  ####################################################### calculate value function (critic)
  
  critic <- trainCritic(allResDiscounted,iteration=i)
  
  ## Advantage: r(s,a) + V(s+1) - V(s)
  #### old R(s) - V(s)
  if(advantageTD) {
    ## NEW: Advantage: r(s,a) + V(s+1) - V(s)
    currResDiscountedAdvantage <- calculateAdvantageKerasCorrect(currResDiscounted, model=critic)
  } else {
    currResDiscountedAdvantage <- calculateAdvantageKeras(currResDiscounted, model=critic) # ## seems to work well?! better than other one. Why? 
  }
  #
  
  currActions <- lapply(currResDiscountedAdvantage, FUN= '[[',3)
  currAdvantages <- lapply(currResDiscountedAdvantage, FUN= '[[',6)
  
  ######## Aggregate duplicate fitness values in archives, sum experience for same individual
  inputArchive <- c(inputArchive,currInputs)
  advantageArchive <- c(advantageArchive,currAdvantages)
  
  annsIds <- sapply(annArchive, FUN= '[[',5)
  repeatID=anyDuplicated(annsIds)
  while(repeatID) {
    agg <- which(annsIds==annsIds[repeatID])
    fitnessArchive[agg[1]] <- mean(fitnessArchive[agg])
    inputArchive[[agg[1]]] <- do.call(rbind,inputArchive[agg])
    advantageArchive[[agg[1]]] <- do.call('c',advantageArchive[agg])
    inputArchive <- inputArchive[-agg[-1]]
    advantageArchive <- advantageArchive[-agg[-1]]
    fitnessArchive <- fitnessArchive[-agg[-1]]
    annArchive <- annArchive[-agg[-1]]
    annsIds <- annsIds[-agg[-1]]
    repeatID=anyDuplicated(annsIds)
  }
  
  ### create model res
  modelRes <- annArchive
  for(j in 1:length(fitnessArchive)) {
    modelRes[[j]]$input <- inputArchive[[j]]
    modelRes[[j]]$advantage <- advantageArchive[[j]]
  }    
  
  ########### EXPERIENCE handling, Fill elite Experience Set
  
  for (j in 1:elitistSelectionSize) {
    if(any(annsFitness[best[j]] >= fitnessElite[j:elitistSelectionSize]) | any(annsFitness[best[j]] >= elitistArchive$fitness)) {
      if(lenArchive < elitistArchiveSize) {
        fitnessElite[j] <- annsFitness[best[j]]
        
        elitistArchive$input <- c(elitistArchive$input,currInputs[best[j]])
        elitistArchive$action <- c(elitistArchive$action,currActions[best[j]])
        elitistArchive$reward <- c(elitistArchive$reward,currRewards[best[j]])
        elitistArchive$advantage <- c(elitistArchive$advantage,currAdvantages[best[j]])
        elitistArchive$fitness <- c(elitistArchive$fitness,annsFitness[best[j]])
        elitistArchive$ann <- c(elitistArchive$ann,anns[best[j]])
        lenArchive=lenArchive + 1
      } else {
        fitnessElite[j] <- annsFitness[best[j]]
        rem <- which.min(elitistArchive$fitness)
        elitistArchive$input[[rem]] <- currInputs[[best[j]]]
        elitistArchive$action[[rem]] <- currActions[[best[j]]]
        elitistArchive$reward[[rem]] <- currRewards[[best[j]]]
        elitistArchive$advantage[[rem]] <- currAdvantages[[best[j]]]
        elitistArchive$fitness[rem] <- annsFitness[best[j]]
        elitistArchive$ann[[rem]] <- anns[[best[j]]]
      }
    }
    
  }
  
  
  ###Recalculate advantag
  
  lenRes <- length(elitistArchive$input)
  for (i in 1:lenRes) {
    reward <- elitistArchive$reward[[i]]
    input <- elitistArchive$input[[i]]
    #print(typeof(input))
    value = critic %>% predict(input)
    valueNextState <- c(value[-1],0)
    elitistArchive$advantage[[i]] <- reward + valueNextState - value 
  }
  
  eliteExperience$input <- do.call(rbind,elitistArchive$input)
  eliteExperience$action <- do.call('c',elitistArchive$action)
  eliteExperience$reward <- do.call('c',elitistArchive$reward)
  eliteExperience$advantage <- do.call('c',elitistArchive$advantage)
  
  #experience <- extractExperienceDiscountedSimple(res, selSize= 0.2)
  ###select at random?
  
  #Ordered by Advantage, take % of best advantage actions to learn
  currLearnExperience <- extractExperienceAdvantageSimple(eliteExperience,numOutputs, selSize= experienceSelectionSize) ## Selection 20% worksquite good
  ## behavior distance, take % of random states to compute distance
  #currModelExperience <- extractExperienceRandom(eliteExperience, selSize= 0.5) ## TODO Testing
  
  selectedExperienceInputs <- currLearnExperience$input
  selectedExperienceAdvantage <- currLearnExperience$advantage
  selectedExperienceOutputs <- currLearnExperience$output
  
  if(diskLogging==TRUE)
  {
    ##main plot
    pdf(paste("plot",instanceName,".pdf",sep=""))
    plot(fitness,main=(paste("Total Evaluations:",length(fitness),
                             "Total Repeats:",totalRepeats,
                             " Elitist Fitness(avg):",mean(eliteRepeats),
                             " Elitist Repeated:",length(eliteRepeats))),ylim=c(10,200))
    abline(h=200, col="red")
    abline(v=initSize, col="blue")
    dev.off()
    ##store data global
    save(fitness,bestMethodCounter,file=paste("result",instanceName,".Rdata",sep=""))
    save(allResDiscounted,file=paste("data",instanceName,".Rdata",sep=""))
  }
  
  ###debugging
  if(nrow(selectedExperienceInputs) != nrow(selectedExperienceOutputs)) {
    print("Wrong Number of input rows! experienceInput")
  }
  if(any(is.na(selectedExperienceInputs))) {
    print("NA in input, Break")
    break
  }
  if(ncol(selectedExperienceInputs) != numInputs) {
    print("Wrong Number of input colums! experienceInput")
  }
  for (debugAnn in modelRes) {
    if(ncol(debugAnn$input) != numInputs) {
      print("Wrong Number of input Colums! modelRes")
      print(debugAnn)
      break
    }
  }
  
  ###DEBUGGING
  #New Behavior Fit, CrossEntropy
  print("Optim Cross Entropy")
  new <- lapply(1:100, function(x) .Call("initChromoR",x*floor(runif(1)*1000),params,funcSet))
  crossEntropyAnns <- c(anns,new) ### include elitist set?!
  crossEntropyAnns <- optimCrossEntropy(crossEntropyAnns,selectedExperienceInputs,selectedExperienceOutputs,selectedExperienceAdvantage,
                                        iterations=1000*trainingFactor,noElitist=2,numberChilds=20,mutationRate=0.05,plotFitness=FALSE)
  crossEntropy <- sapply(crossEntropyAnns,function(ann) computeCrossEntropyLossWeighted(x=NULL,ann,selectedExperienceInputs,selectedExperienceOutputs,selectedExperienceAdvantage))
  crossEntropyAnns <- crossEntropyAnns[order(crossEntropy)]
  print(paste("CrossEntropy",min((crossEntropy))))
  eliteCrossEntropyAnn <- crossEntropyAnns[[1]]
  
  
  #New Behavior Fit, Fitness distance
  print("Optim Absolute BehFitness")
  new <- lapply(1:100, function(x) .Call("initChromoR",x*floor(runif(1)*1000),params,funcSet))
  behFitAnns <- c(anns,new) ### include elitist set?!
  behFitAnns <- optimAbsBehaviour(behFitAnns,selectedExperienceInputs,selectedExperienceOutputs,selectedExperienceAdvantage,
                                  iterations=1000*trainingFactor,noElitist=2,numberChilds=20,mutationRate=0.05,plotFitness=FALSE)
  behFit <- sapply(behFitAnns,function(ann) compareBehaviorWeightedReward(x=NULL,ann,selectedExperienceInputs,selectedExperienceOutputs,selectedExperienceAdvantage))
  behFitAnns <- behFitAnns[order(behFit)]
  print(paste("Abs Bev Fit",min((behFit))))
  eliteAbsBehAnn <- behFitAnns[[1]]
  
  ##new SurrOptim
  print("SMB-NE")
  
  behaviorDistanceKriging <- function(annA,annB,input=selectedExperienceInputs, reward=selectedExperienceAdvantage) {
    outputA <- queryNetworkSoftmax(annA,input)
    outputB <- queryNetworkSoftmax(annB,input)
    res <- abs(outputA - outputB)*reward
    mean(res,na.rm=TRUE/mean(abs(reward))) ###TODO NA?
  }
  
  behaviorDistanceKrigingOneToOne <- function(annResA,annResB) {
    input <- rbind(annResA$input,annResB$input)
    #advantage <- c(annResA$advantage,annResB$advantage)
    outputA <- queryNetworkSoftmax(annResA,input)
    outputB <- queryNetworkSoftmax(annResB,input)
    res <- abs(outputA - outputB)#*advantage
    #mean(res,na.rm=TRUE/mean(abs(advantage))) ###TODO NA?
    mean(res,na.rm=TRUE)
  }
  
  ### elite inputs, same for all
  #fit <- modelKriging(annArchive,-fitnessArchive,behaviorDistanceKriging,control=list(algThetaControl=list(method="NLOPT_GN_DIRECT_L"),reltol=1e-16,nevals=200,useLambda=T,reinterpolate=F,scaling=T,combineDistances=T)) ###Reinterpoltate für EI an/sonst aus
  behSurrogate <- modelKriging(modelRes,-fitnessArchive,behaviorDistanceKrigingOneToOne,control=list(algThetaControl=list(method="NLOPT_GN_DIRECT_L"),reltol=1e-16,nevals=500,useLambda=T,reinterpolate=F,scaling=T,combineDistances=T)) ###Reinterpoltate für EI an/sonst aus
  new <- lapply(1:1000, function(x) .Call("initChromoR",x*floor(runif(1)*1000),params,funcSet))
  pop <- c(new,behFitAnns,crossEntropyAnns) ### include elitist set?!
  surrFit <- optimKriging(pop,behSurrogate,fitnessArchive,iterations=500*trainingFactor,noElitist=2,numberChilds=8,mutationRate=0.01)
  bestSurrAnn <- surrFit[[1]]
  ##elite mutates
  mutateElite <- mutateWrapper(eliteFitAnn,1337,mutationRate = 0.01)
  
  ##next Generation
  anns <- list(eliteFitAnn,eliteCrossEntropyAnn,eliteAbsBehAnn,bestSurrAnn,mutateElite)
  
  ###Watch for Duplicates / 
  dupli= anyDuplicated(anns)
  while (dupli) {
    print("duplicates!!")
    print(duplicated(anns))
    anns[[dupli]] <- mutateWrapper(anns[[dupli]],1337,mutationRate = 0.01)
    dupli= anyDuplicated(anns)
  }
  
  for (id in 1:(popSize-1)) {
    anns[[id+1]]$id <- (initSize+(i-1)*popSize)+id
  }
  
  #write.csv(resultsAll, file = paste("resCartPole/resultsSMBNE_PREranMutlargeDebug2",Sys.getenv("PBS_ARRAYID"),".csv",sep=""))
  #write.csv(resultsY, file = paste("resCartPole/yRes/resultsSMBNE_PREranMutlargeDebug2_y",Sys.getenv("PBS_ARRAYID"),".csv",sep=""))
}

logData <- data.frame(seed=integer(1), 
                      problem=character(1), 
                      params=character(1),
                      nEvaluations=integer(1), 
                      yBest=numeric(1), 
                      timeSteps=numeric(1),
                      bestMethod=numeric(1),
                      stringsAsFactors = FALSE)

t <- table(bestMethodCounter)

logData[1,1] <- seed
logData[1,2] <- "CartPole"
logData[1,3] <- "1.0"
logData[1,4] <- length(fitness)
logData[1,5] <- max(fitness)
logData[1,6] <- sum(fitness)
logData[1,7] <- names(t)[which.max(t)]
logData[1,8:(length(fitness)+7)] <- fitness

write.csv(logData, file = paste("resCartPole/bnet_cartpole_t1_",seed,".csv",sep=""))
save(allResDiscountedValue, file= paste("resCartPole/bnet_cartpole_t1_res_",Sys.getenv("PBS_ARRAYID"),".Rdata",sep=""))
