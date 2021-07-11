library(reticulate)
#library(SPOT)
library(pbapply)
library(emoa)
#library(parallel)
library(CEGO)
library(randomForest)
#library(ecr)


setwd("~/Google Drive/Work/beNEMOresults")
source("~/Google Drive/Work/Stor2020c/code/beNEMO_functions.R")

use_python("/Library/Frameworks/Python.framework/Versions/3.7/bin/python3", required = TRUE) #local
#use_python("/usr/bin/python3.6", required=TRUE) #server
dyn.load("cgp_R.so")
Rgym <- import("gym")


##function options, experimental design
bootstrapCritic=FALSE
advantageCorrect=TRUE
experienceSelectionSize <- 0.2

##Logging
diskLogging=TRUE
logID <- floor(runif(1)*1000)
instanceName=paste("test",logID,sep="")

env = Rgym$make('MountainCar-v0')
env$reset()

##params
##Fitness function max=1, min=-1
fitSign=1

numInputs <- 2 #70input+1Bias
numOutputs <- 3
numSamples <- 1
funcSet <- c("tanh","sig","gauss","step")
params <- c(numInputs,numNodes=200,numOutputs,nodeArity=10)
#criticParams <- c(numInputs=4,numNodes=200,numOutputs=1,nodeArity=10)
popSize <- 6
initSize <- popSize*2
elitistSelectionSize <- popSize
elitistArchiveSize <- popSize*2

###Iterative Variables init
fitness <- NULL

##elitists
eliteExperience <- NULL
elitistArchive <- NULL
fitnessElite <- rep(-300,elitistSelectionSize)
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
allTesting <- NULL


####reConfig for MC Problem
computeFitnessStates <- function(x,steps,instances, stochasticPolicy=TRUE,fitSign=1) {
  sum <- 0
  states= NULL
  obs_R <- list()
  for(j in 1:instances) {
    episode_rewards = 0
    #fixed seed for no noise, python seed has to be set additionally to R, cause they are independet
    #set.seed(123)
    #py_set_seed(123, disable_hash_randomization = TRUE)
    obs = env$reset()
    outputActions = NULL
    rewards = NULL
    obs_R[[1]] <- obs
    action =  rep(1L,x$meta[3])
    states <- rbind(states,as.vector(obs_R[[1]]))
    #states <- rbind(states,c(1,as.vector(obs_R[[1]])))
    for (i in 1:steps) {
      if(stochasticPolicy) {
        action = softmaxNavigation_policy_stochastic(x, obs_R)
      } else {
        action = softmaxNavigation_policy(x, obs_R)
      }
      obs_R= env$step(action)
      
      #if (i > 1) {
      #  lastAction= as.integer(tail(outputActions,n=1))
      #  if(action==0 && lastAction==1)
      #  {penalty=-1}
      #  if(action==1 && lastAction==0)
      #  {penalty=-1}
      #  if(action==2 && lastAction==3)
      #  {penalty=-1}
      #  if(action==3 && lastAction==2) 
      #  {penalty=-1}
      #}
      episode_rewards= episode_rewards + obs_R[[2]][1] ##normal
      #store state vector (at the moment last)
      
      outputActions <- c(outputActions,action)   
      rewards <- c(rewards, obs_R[[2]][1])
      if (obs_R[[3]]==TRUE | i==steps) {
        #print("DONE iterations:")
        #print(i)
        sum= i
        ###fun fun fun
        #rewards[i] <- i
        ######
        retValue <- list(epReward=episode_rewards*fitSign,input=states,action=outputActions,reward=rewards)
        env$close()
        return(retValue)
      } else {
        states <- rbind(states,c(as.vector(obs_R[[1]]))) ##icl. Bias # final state not stores (since there is no action or reward)
      }
    }
    sum= (sum + episode_rewards)*fitSign
  }
  retValue <- list(epReward=sum/instances,input=states,action=outputActions,reward=rewards)
  return(retValue)
}

###INIT of first gen ANNS  (OFFLINE possible, TODO)
load("mcarExp.Rdata")

learnFromOfflineExperienceMcar <- function(allRes,elitistArchiveSize)
{
  disReward <- lapply(allRes, FUN= '[[',5) ###mcar
  fitness <- sapply(disReward, min)
  orderExp <- order(fitness,decreasing = TRUE)
  allRes <- allRes[orderExp][1:50]
  
  #print("a")
  allInputs <- lapply(allRes, FUN= '[[',2)
  allRewards <- lapply(allRes, FUN= '[[',5)
  
  learnInputs <- do.call(rbind,allInputs)
  learnValues <- do.call('c',allRewards)
  #print("b")
  testdata <- data.frame(learnInputs,learnValues)
  forest <- randomForest(learnValues ~ ., data=testdata,ntree=500)
  #print("c")
  allResDiscountedAdvantage <- calculateAdvantageCritic(allRes,forest)

  championRes <- allResDiscountedAdvantage[1:elitistArchiveSize]
  #print("d")
  #orderExp <- order(fitness,decreasing = TRUE)
  allRes <- allRes[orderExp]
  
  expInputs <- lapply(championRes, FUN= '[[',2)
  expActions <- lapply(championRes, FUN= '[[',3)
  expReward <- lapply(championRes, FUN= '[[',5)
  expAdvantages <- lapply(championRes, FUN= '[[',6)
  
  eliteExperience$input <- do.call(rbind,expInputs)
  eliteExperience$action <- do.call('c',expActions)
  eliteExperience$reward <- do.call('c',expReward)
  eliteExperience$advantage <- do.call('c',expAdvantages)
  #print("e")
  #Ordered by Advantage, take % of best advantage actions to learn
  currLearnExperience <- extractExperienceAdvantageSimple(eliteExperience,numOutputs, selSize= experienceSelectionSize)
  #print("f")
  goodExperienceInputs <- currLearnExperience$input
  goodExperienceReward <- currLearnExperience$advantage
  goodExperienceOutputs <- currLearnExperience$output
  
  numInPop <- 2
  pop <- NULL
  for (j in 1:numInPop) {
  print("Optim Cross Entropy")
  new <- lapply(1:100, function(x) .Call("initChromoR",x*floor(runif(1)*1000),params,funcSet))
  crossEntropyAnns <- c(new) ### include elitist set?!
  crossEntropyAnns <- optimCrossEntropy(crossEntropyAnns,goodExperienceInputs,goodExperienceOutputs,goodExperienceReward,iterations=1000,plotFitness=FALSE)
  crossEntropy <- sapply(crossEntropyAnns,function(ann) computeCrossEntropyLossWeighted(x=NULL,ann,goodExperienceInputs,goodExperienceOutputs,goodExperienceReward))
  crossEntropyAnns <- crossEntropyAnns[order(crossEntropy)]
  print(paste("CrossEntropy",min((crossEntropy))))
  pop <- c(pop, list(crossEntropyAnns[[order(crossEntropy)[1]]]))
  
  #New Behavior Fit, Fitness distance
  print("Optim Absolute BehFitness")
  new <- lapply(1:100, function(x) .Call("initChromoR",x*floor(runif(1)*1000),params,funcSet))
  behFitAnns <- c(new) ### include elitist set?!
  behFitAnns <- optimAbsBehaviour(behFitAnns,goodExperienceInputs,goodExperienceOutputs,goodExperienceReward,iterations=1000,plotFitness=FALSE)
  behFit <- sapply(behFitAnns,function(ann) compareBehaviorWeightedReward(x=NULL,ann,goodExperienceInputs,goodExperienceOutputs,goodExperienceReward))
  behFitAnns <- behFitAnns[order(behFit)]
  print(paste("Abs Bev Fit",min((behFit))))
  pop <- c(pop, list(behFitAnns[[order(behFit)[1]]]))
  }
  
  currLearnExperience <- extractExperienceAdvantageSimple(eliteExperience,numOutputs, selSize= 1)
  
  goodExperienceInputs <- currLearnExperience$input
  goodExperienceReward <- currLearnExperience$advantage
  goodExperienceOutputs <- currLearnExperience$output
  
  numInPop <- 2
  for (j in 1:numInPop) {
    print("Optim Cross Entropy")
    new <- lapply(1:100, function(x) .Call("initChromoR",x*floor(runif(1)*1000),params,funcSet))
    crossEntropyAnns <- c(new) ### include elitist set?!
    crossEntropyAnns <- optimCrossEntropy(crossEntropyAnns,goodExperienceInputs,goodExperienceOutputs,goodExperienceReward,iterations=1000,plotFitness=FALSE)
    crossEntropy <- sapply(crossEntropyAnns,function(ann) computeCrossEntropyLossWeighted(x=NULL,ann,goodExperienceInputs,goodExperienceOutputs,goodExperienceReward))
    crossEntropyAnns <- crossEntropyAnns[order(crossEntropy)]
    print(paste("CrossEntropy",min((crossEntropy))))
    pop <- c(pop, list(crossEntropyAnns[[order(crossEntropy)[1]]]))
    
    #New Behavior Fit, Fitness distance
    print("Optim Absolute BehFitness")
    new <- lapply(1:100, function(x) .Call("initChromoR",x*floor(runif(1)*1000),params,funcSet))
    behFitAnns <- c(new) ### include elitist set?!
    behFitAnns <- optimAbsBehaviour(behFitAnns,goodExperienceInputs,goodExperienceOutputs,goodExperienceReward,iterations=1000,plotFitness=FALSE)
    behFit <- sapply(behFitAnns,function(ann) compareBehaviorWeightedReward(x=NULL,ann,goodExperienceInputs,goodExperienceOutputs,goodExperienceReward))
    behFitAnns <- behFitAnns[order(behFit)]
    print(paste("Abs Bev Fit",min((behFit))))
    pop <- c(pop, list(behFitAnns[[order(behFit)[1]]]))
  }
  return(pop)
}

offlineExperience <- NULL
offlineExperience <- allResDiscounted
if(is.null(offlineExperience)) {
  anns <- lapply(1:initSize, function(x) .Call("initChromoR",x*floor(runif(1)*1000),params,funcSet))
  for (id in 1:initSize) {
    anns[[id]]$id <- id
  }
} else {
  anns <- learnFromOfflineExperienceMcar(offlineExperience,elitistArchiveSize)
  allResDiscounted <- NULL ### include in set? 
  for (id in 1:length(anns)) {
    anns[[id]]$id <- id
  }
}
currRes <- lapply(anns,computeFitnessStates,steps=200,instances=1,stochasticPolicy=TRUE,fitSign)

###Function LOOP
for (i in 1:30) {
  print(paste("Iteration:",i))
  ####### Testing and generating EXPERIENCE, use stochastic policy for init gen
  if(i>1) {
    currRes <- lapply(anns,computeFitnessStates,steps=500,instances=1,stochasticPolicy=FALSE,fitSign)
  }
  
  #Testing
  #if(i>1) {
  #  eliteRes <- computeFitnessStates(anns[[1]],steps=5000,instances=1,stochasticPolicy=TRUE)
  #  currRes <- lapply(anns[-1],computeFitnessStates,steps=5000,instances=1,stochasticPolicy=FALSE)
  #  currRes <- c(list(eliteRes),currRes)
  #}
  
  ###### FIND Robust Elitist / do Repeats #### 
  annsFitness <- sapply(currRes, FUN= '[[',1) ## for pendulum (all rewards=1)
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
        ##max repeats=10
        len <- min(c(len,6))
        challengerFitness <- annsFitness[challenger[j]]
        if(challengerFitness <= mean(eliteRepeats)) {
          break
        }
        if (len < 3) {
          #eliteRepeats <- c(eliteRepeats,testInstance(eliteFitAnn,instances = 1,stochastic=FALSE))  
          eliteRes <- computeFitnessStates(eliteFitAnn,steps=500,instances=1,stochasticPolicy=FALSE,fitSign)
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
          challengerRes <- computeFitnessStates(anns[[challenger[j]]],steps=500,instances=1,stochasticPolicy=FALSE,fitSign)
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
                           " Elitist Repeated:",length(eliteRepeats))),ylim=c(-200,0))
  abline(h=-110, col="red")
  abline(v=initSize, col="blue")
  
  ####################################################### calculate value function (critic)
  
  
  
  ###### TODO: Fitting Q(s,a) -> not Value! Q: r(s,a) + V(s+1)
  ####  Correct?   Bootstrapped value function 
  
  ###Selection Window
  selWindow <- max(1,length(allResDiscounted)-49):length(allResDiscounted)
  
  allResInputs <- lapply(allResDiscounted[selWindow], FUN= '[[',2)
  
  if(i>1 & bootstrapCritic==TRUE) {  ##first iteration is init 
    allResDiscountedValue <- oneStepValue(allResDiscounted[selWindow], critic=forest)
    allValues <- lapply(allResDiscountedValue, FUN= '[[',6)
  } else {
    allValues <- lapply(allResDiscounted[selWindow], FUN= '[[',5) ## init V(s) to R(s)
  }
  
  learnInputs <- do.call(rbind,allResInputs)
  learnValues  <- do.call('c',allValues)
  
  testdata <- data.frame(learnInputs,learnValues)
  forest <- randomForest(learnValues ~ ., data=testdata,ntree=500)
  #test <- as.vector(predict(forest,data.frame(valueInputs)))
  #test - valueExpRewards 
  #currResDiscounted <- calculateAdvantageRes(currResDiscounted)
  
  #### TODO:
  ## Advantage: r(s,a) + V(s+1) - V(s)
  
  
  #### old R(s) - V(s)
  if(advantageCorrect) {
    ## NEW: Advantage: r(s,a) + V(s+1) - V(s)
    currResDiscountedAdvantage <- calculateAdvantageCriticCorrect(currResDiscounted,critic=forest)
  } else {
    currResDiscountedAdvantage <- calculateAdvantageCritic(currResDiscounted,critic=forest) # ## seems to work well?! better than other one. Why? 
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
    advantageArchive[[agg[1]]] <- do.call(c,advantageArchive[agg])
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
  
  eliteExperience$input <- do.call(rbind,elitistArchive$input)
  eliteExperience$action <- do.call('c',elitistArchive$action)
  eliteExperience$reward <- do.call('c',elitistArchive$reward)
  eliteExperience$advantage <- do.call('c',elitistArchive$advantage)
  
  #experience <- extractExperienceDiscountedSimple(res, selSize= 0.2)
  ###select at random?
  
  #Ordered by Advantage, take % of best advantage actions to learn
  currLearnExperience <- extractExperienceAdvantageSimple(eliteExperience,numOutputs, selSize= experienceSelectionSize) ## Selection 20% worksquite good
  ## behavior distance, take % of random states to compute distance
  currModelExperience <- extractExperienceRandom(eliteExperience, selSize= 0.5) ## TODO Testing
  
  goodExperienceInputs <- currLearnExperience$input
  goodExperienceReward <- currLearnExperience$advantage
  goodExperienceOutputs <- currLearnExperience$output
  
  currLearnExperienceTest <- extractExperienceRewardSimple(eliteExperience,numOutputs, selSize= experienceSelectionSize)
  
  if(diskLogging==TRUE)
  {
    ##main plot
    pdf(paste("plot",instanceName,".pdf",sep=""))
    plot(fitness,main=(paste("Total Evaluations:",length(fitness),
                             "Total Repeats:",totalRepeats,
                             " Elitist Fitness(avg):",mean(eliteRepeats),
                             " Elitist Repeated:",length(eliteRepeats))),ylim=c(-1,1))
    abline(h=-200, col="red")
    abline(v=initSize, col="blue")
    dev.off()
    ##store data global
    save(fitness,bestMethodCounter,file=paste("result",instanceName,".Rdata",sep=""))
    save(allResDiscounted,file=paste("data",instanceName,".Rdata",sep=""))
  }
  
  ###debugging
  if(nrow(goodExperienceInputs) != nrow(goodExperienceOutputs)) {
    print("Wrong Number of input rows! experienceInput")
  }
  if(any(is.na(goodExperienceInputs))) {
    print("NA in input, Break")
    break
  }
  if(ncol(goodExperienceInputs) != numInputs) {
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
  new <- lapply(1:90, function(x) .Call("initChromoR",x*floor(runif(1)*1000),params,funcSet))
  crossEntropyAnns <- c(anns,new,elitistArchive$ann) ### include elitist set?!
  crossEntropyAnns <- optimCrossEntropy(crossEntropyAnns,goodExperienceInputs,goodExperienceOutputs,goodExperienceReward,iterations=1000,plotFitness=FALSE)
  crossEntropy <- sapply(crossEntropyAnns,function(ann) computeCrossEntropyLossWeighted(x=NULL,ann,goodExperienceInputs,goodExperienceOutputs,goodExperienceReward))
  crossEntropyAnns <- crossEntropyAnns[order(crossEntropy)]
  print(paste("CrossEntropy",min((crossEntropy))))
  eliteCrossEntropyAnn <- crossEntropyAnns[[order(crossEntropy)[1]]]
  
  
  #New Behavior Fit, Fitness distance
  print("Optim Absolute BehFitness")
  new <- lapply(1:90, function(x) .Call("initChromoR",x*floor(runif(1)*1000),params,funcSet))
  behFitAnns <- c(anns,new,elitistArchive$ann) ### include elitist set?!
  behFitAnns <- optimAbsBehaviour(behFitAnns,goodExperienceInputs,goodExperienceOutputs,goodExperienceReward,iterations=1000,plotFitness=FALSE)
  behFit <- sapply(behFitAnns,function(ann) compareBehaviorWeightedReward(x=NULL,ann,goodExperienceInputs,goodExperienceOutputs,goodExperienceReward))
  behFitAnns <- behFitAnns[order(behFit)]
  print(paste("Abs Bev Fit",min((behFit))))
  eliteAbsBehAnn <- behFitAnns[[order(behFit)[1]]]
  
  #New Behavior Fit, CrossEntropy
  print("Optim Absolute BehFitness, Reward")
  new <- lapply(1:90, function(x) .Call("initChromoR",x*floor(runif(1)*1000),params,funcSet))
  behFitRewardAnns <- c(anns,new,elitistArchive$ann) ### include elitist set?!
  behFitRewardAnns <- optimAbsBehaviour(behFitAnns,currLearnExperienceTest$input,currLearnExperienceTest$output,currLearnExperienceTest$reward,iterations=1000,plotFitness=FALSE)
  behFitReward <- sapply(behFitRewardAnns,function(ann) compareBehaviorWeightedReward(x=NULL,ann,currLearnExperienceTest$input,currLearnExperienceTest$output,currLearnExperienceTest$reward))
  behFitRewardAnns <- behFitRewardAnns[order(behFitReward)]
  print(paste("behFitReward",min((behFitReward))))
  eliteBehFitRewardAnn <- behFitRewardAnns[[order(behFitReward)[1]]]
  
  ##new SurrOptim
  print("SMB-NE")
  
  behaviorDistanceKriging <- function(annA,annB,input=currModelExperience$input, reward=currModelExperience$advantage) {
    outputA <- queryNetworkSoftmax(annA,input)
    outputB <- queryNetworkSoftmax(annB,input)
    res <- abs(outputA - outputB)*abs(reward)
    mean(res,na.rm=TRUE/mean(abs(reward))) ###TODO NA?
  }
  
  behaviorDistanceKrigingOneToOne <- function(annResA,annResB) {
    input <- rbind(annResA$input,annResB$input)
    advantage <- c(annResA$advantage,annResB$advantage)
    outputA <- queryNetworkSoftmax(annResA,input)
    outputB <- queryNetworkSoftmax(annResB,input)
    res <- abs(outputA - outputB)*abs(advantage)
    mean(res,na.rm=TRUE/mean(abs(advantage))) ###TODO NA?
    #mean(res,na.rm=TRUE)
  }
  
  ### elite inputs, same for all
  #fit <- modelKriging(annArchive,-fitnessArchive,behaviorDistanceKriging,control=list(algThetaControl=list(method="NLOPT_GN_DIRECT_L"),reltol=1e-16,nevals=200,useLambda=T,reinterpolate=F,scaling=T,combineDistances=T)) ###Reinterpoltate für EI an/sonst aus
  
  fit <- modelKriging(modelRes,-fitnessArchive,behaviorDistanceKrigingOneToOne,control=list(algThetaControl=list(method="NLOPT_GN_DIRECT_L"),reltol=1e-16,nevals=200,useLambda=T,reinterpolate=F,scaling=T,combineDistances=T)) ###Reinterpoltate für EI an/sonst aus
  new <- lapply(1:1000, function(x) .Call("initChromoR",x*floor(runif(1)*1000),params,funcSet))
  pop <- c(elitistArchive$ann,new) ### include elitist set?!
  predictedMean <- fpred(pop,fit,min(-fitnessArchive))
  bestExpAnn <- pop[[order(predictedMean)[1]]]
  bestEI <- predictedMean[order(predictedMean)[1]]
  for (j in 1:100) {
    offspring <- lapply(1:4,function(x) mutateWrapper(ann=bestExpAnn,seed=123,mutationRate=0.05))
    predictedMean <- fpred(offspring,fit,min(-fitnessArchive))
    bestOffspringEI <- order(predictedMean)[1]
    if(bestOffspringEI <= bestEI) {
      bestExpAnn <- offspring[[bestOffspringEI]]
      bestEI <- predictedMean[bestOffspringEI]
    }
  }
  print(paste("SMBNE Predicted Mean:",bestEI))
  
  ##elite mutates
  mutateElite <- mutateWrapper(eliteFitAnn,1337,mutationRate = 0.01)
  
  ##next Generation
  anns <- list(eliteFitAnn,eliteCrossEntropyAnn,eliteAbsBehAnn,eliteBehFitRewardAnn,bestExpAnn,mutateElite)
  
  ###Watch for Duplicates / 
  dupli= anyDuplicated(anns)
  while (dupli) {
    print("duplicates!!")
    print(duplicated(anns))
    anns[[dupli]] <- mutateWrapper(anns[[dupli]],1337,mutationRate = 0.01)
    dupli= anyDuplicated(anns)
  }
  
  for (id in 1:5) {
    anns[[id+1]]$id <- (initSize+(i-1)*popSize)+id
  }
  print("Testing Stopping Criteria")
  testing <- sapply(anns,testInstance,steps=500,instances=100,fitSign=fitSign)
  print(testing)
  bestMethodCounter <- c(bestMethodCounter,which.max(testing))
  print("Best Method Counter:")
  print(bestMethodCounter)
  allTesting <- c(allTesting,testing)
  if(any(testing >= -110 )) {
   print("Found Solution")
  break
  }
}

len <- 1

seed <- 
problem <- "MCar Offline"
population <- "E CE BF1 BF2 SBP M"
setup <- "SelSize 1"
iterations <- length(fitness)
bestFitnessFound <- max(fitness)
bestFitnessTest <- max(allTesting)



resultsAll <- data.frame(seed,problem,population,setup,iterations,bestFitness,stringsAsFactors = FALSE)


#resultsAll[1,8:(length(y)+7)] <- y  
#fitness
#bestMethodCounter
#allTesting

write.csv(resultsAll, file = paste("resCartPole/resultsSMBNE_PREranMutlargeDebug2",Sys.getenv("PBS_ARRAYID"),".csv",sep=""))





