library(reticulate)
#library(SPOT)
library(pbapply)
library(parallel)
library(CEGO)
library(randomForest)
#library(ecr)

#setwd("~/Google Drive/Work/beNEMOresults")
#use_python("/Library/Frameworks/Python.framework/Versions/3.7/bin/python3", required = TRUE) #local
use_python("/usr/bin/python3.8", required=TRUE) #server
dyn.load("cgp_R.so")
Rgym <- import("gym")

####EXPERIEMENT Params
##function options, experimental design
advantageCorrect=FALSE  #a t/f
bootstrapCritic=FALSE  # b t/f
experienceSelectionSize <- 0.2

seed <- as.integer(Sys.getenv("PBS_ARRAYID")) #server
if(is.na(seed)) seed <- 12345
#seed <- 123435 # localtest
set.seed(seed)


##Logging
diskLogging=TRUE
instanceName=paste("experiment",seed)
dirName="afbf02"

env = Rgym$make('CartPole-v1')
env$reset()

mutateWrapper <- function(ann,seed,mutationRate=0.05) {
  seed <- floor(runif(1)*1000000*seed)
  ### safety call
  ann <- ann[c(1,2,3,4)]
  #chromo <- .Call("mutateR",chromo,seed,mutationRate) #Goldman Mutation
  ann <- .Call("mutateR_Random",ann,seed,mutationRate) #Random Mutation
}

mutateWrapperOffspring <- function(x,ann,seed,mutationRate=0.05) {
  seed <- floor(runif(1)*1000000*seed)
  #chromo <- .Call("mutateR",chromo,seed,mutationRate) #Goldman Mutation
  ### safety call
  ann <- ann[c(1,2,3,4)]
  ann <- .Call("mutateR_Random",ann,seed,mutationRate) #Random Mutation
}

queryNetwork <- function(ann, input) {
  #input <- as.matrix(input)
  len <- nrow(input)
  dim <- ncol(input)
  #print(len)
  if (is.null(len)) {
    len <- 1
    dim <- length(input)
  }
  if(dim != ann$meta[1]) {
    print("WRONG input dim, query network failed")
    break
  }
  #input <- cbind(1,input)
  #print(dim)
  #print(input)
  #testing <- apply(input,1,sum)
  #print(testing)
  ann <- ann[c(1,2,3,4)]
  output <- .Call("R_computeOutput", ann, dim, len, as.vector(t(input)))
  ### Impotant bug workaround, research why this is happening!!
  output <- matrix(as.vector(output), nrow = len, byrow = TRUE)
  #output2 <- apply(input,1,function(x) .Call("R_computeOutput",net,dim,1,x))
  #output <- unlist(output)
  #print(t(output2))
  return(output)
}
###stable version of softmax!
softmax <- function(x) {
  z= x-max(x)
  exp(z)/sum(exp(z))
}

queryNetworkSoftmax <- function(ann, input) {
  #input <- as.matrix(input)
  len <- nrow(input)
  dim <- ncol(input)
  #print(len)
  if (is.null(len)) {
    len <- 1
    dim <- length(input)
  }
  if(dim != ann$meta[1]) {
    print("WRONG input dim, query network failed")
    break
  }
  
  ann <- ann[c(1,2,3,4)]
  output <- .Call("R_computeOutput", ann, dim, len, as.vector(t(input)))
  ### Impotant bug workaround, research why this is happening!!
  output <- matrix(as.vector(output), nrow = len, byrow = TRUE)
  #print(output)
  return(t(apply(output,1,softmax)))
}

behaviorDistanceKriging <- function(annA,annB,input) {
  outputA <- queryNetworkSoftmax(annA,input)
  outputB <- queryNetworkSoftmax(annB,input)
  res <- abs(outputA - outputB)
  mean(res,na.rm=TRUE) ###TODO NA?
}

compareBehavior <- function(x,ann,inputBeh,outputBeh) {
  if(!is.null(x)) {
    activeNodes <- ann$nodes[ann$nodes[,1]==1,]
    activeNodes[,-c(1:2,seq(3,ncol(activeNodes),2))] <- x
    ann$nodes[ann$nodes[,1]==1,] <- activeNodes
  }
  input <- inputBeh
  output <- queryNetworkSoftmax(ann,input)
  #print(output)
  res <- abs(output - outputBeh)
  #print(mean(res))
  mean(res,na.rm=TRUE) ###TODO NA?
}

compareBehaviorLog <- function(x,ann,inputBeh,outputBeh) {
  if(!is.null(x)) {
    activeNodes <- ann$nodes[ann$nodes[,1]==1,]
    activeNodes[,-c(1:2,seq(3,ncol(activeNodes),2))] <- x
    ann$nodes[ann$nodes[,1]==1,] <- activeNodes
  }
  input <- inputBeh
  output <- queryNetworkSoftmax(ann,input)
  res <- 1/log(abs(output - outputBeh))
  #print(mean(res))
  return(max(-res) + mean(-res))
}

countNodes <- function(ann) {
  sum(ann$nodes[,1]==1)
}

behaviorDistance <- function(otherBehavior,ownBehavior) {
  mean(abs(ownBehavior-otherBehavior))
}

actionsToOutput <- function(experienceAction) {
  output<- c(0,0) ####number hardoced!!! TODO
  output[experienceAction+1] <- 1/length(experienceAction) ###hä?
  return(output)
} 

actionsToAvoid <- function(experienceAction) {
  output<- c(0,0) ####number hardoced!!! TODO
  output[experienceAction+1] <- 1
  return(output)
} 

extractExperience <- function(res) {
  input <- res$input
  action <- res$action
  reward <- res$reward
  uniqueInputs <- unique(input)
  len <- nrow(uniqueInputs)
  expInput <- NULL
  expPosOutput <- NULL
  expNegOutput <- NULL
  for(i in 1:len) {
    id <- which(apply(input, 1, function(x) identical(x, uniqueInputs[i,])))
    expInput <- rbind(expInput,input[id[1],])
    expActions <- action[id]
    expReward <- reward[id]
    selPosAction <- unique(expActions[which(expReward >= -0.01)])
    selNegAction <- unique(expActions[which(expReward < -0.01)])
    expPosOutput<- rbind(expPosOutput,actionsToOutput(selPosAction))
    expNegOutput<- rbind(expNegOutput,actionsToAvoid(selNegAction))
  }
  return(list(input=expInput,posOutput=expPosOutput,negOutput=expNegOutput))
}

extractExperienceDiscounted <- function(res) {
  input <- res$input
  action <- res$action
  reward <- res$reward
  uniqueInputs <- unique(input)
  len <- nrow(uniqueInputs)
  expInput <- NULL
  expPosOutput <- NULL
  expNegOutput <- NULL
  expRewardArchive <- NULL
  for(i in 1:len) {
    id <- which(apply(input, 1, function(x) identical(x, uniqueInputs[i,])))
    expInput <- rbind(expInput,input[id[1],])
    expActions <- action[id]
    expRewards <- reward[id]
    #aggregate Rewards
    agg <- data.frame(expActions,expRewards)
    agg <- aggregate(agg,by=(list(agg$expActions)),FUN=mean)
    expActions <- agg$expActions
    expRewards <- agg$expRewards
    expRewardArchive <- c(expRewardArchive,expRewards)
    #Select and sort
    selPosAction <- expActions[which(expRewards >= -0.99)]
    selPosRewards <- expRewards[which(expRewards >= -0.99)]
    probs <- softmax(selPosRewards)
    expPosOutputNew<- c(0,0) ####number hardoced!!! TODO
    expPosOutputNew[selPosAction+1] <- probs
    expPosOutput<- rbind(expPosOutput,expPosOutputNew)
    selNegAction <- expActions[which(expRewards < -0.99)]
    expNegOutput<- rbind(expNegOutput,actionsToAvoid(selNegAction))
  }
  return(list(input=expInput,posOutput=expPosOutput,negOutput=expNegOutput,rewardArchive=expRewardArchive))
}

extractExperienceRewardSimple <- function(res,selSize= 0.2) {
  input <- res$input[order(res$reward),]
  action <- res$action[order(res$reward)]
  reward <- res$reward[order(res$reward)]
  len <- length(reward)
  sel <- len * selSize
  from <- len - sel
  expInput <- input[from:len,]
  expRewardArchive <- reward[from:len]
  expPosAction <- action[from:len]
  expPosOutput <- t(sapply(expPosAction,actionsToOutput))
  expNegOutput=NULL
  return(list(input=expInput,posOutput=expPosOutput,negOutput=expNegOutput,rewardArchive=expRewardArchive))
}

extractExperienceAdvantageSimple <- function(res,selSize= 0.2) {
  input <- res$input[order(res$advantage),]
  action <- res$action[order(res$advantage)]
  advantage <- res$advantage[order(res$advantage)]
  len <- length(advantage)
  sel <- len * selSize
  from <- len - sel
  expInput <- input[from:len,]
  expAdvantage <- advantage[from:len]
  expPosAction <- action[from:len]
  expPosOutput <- t(sapply(expPosAction,actionsToOutput))
  return(list(input=expInput,output=expPosOutput,advantage=expAdvantage))
}

extractExperienceRandom <- function(res,selSize= 0.2) {
  input <- res$input[order(res$advantage),]
  action <- res$action[order(res$advantage)]
  advantage <- res$advantage[order(res$advantage)]
  len <- length(advantage)
  numSel <- len * selSize
  sel <- sample(1:len,numSel)
  expInput <- input[sel,]
  expAdvantage <- advantage[sel]
  expPosAction <- action[sel]
  expPosOutput <- t(sapply(expPosAction,actionsToOutput))
  return(list(input=expInput,output=expPosOutput,advantage=expAdvantage))
}

discountRewards <- function(reward,discountFactor=0.99) {
  disReward <- NULL
  seqReward <- 0
  len <- length(reward)
  for (i in len:1) {
    seqReward <- seqReward*discountFactor + reward[i]
    disReward[i] <- seqReward
  }
  return(disReward)
}

discountRewardsPendulum <- function(reward,discountFactor=0.99) {
  disReward <- NULL
  seqReward <- 0
  len <- length(reward)
  reward[len] <- len - 195
  for (i in len:1) {
    seqReward <- seqReward*discountFactor + reward[i]
    disReward[i] <- seqReward
  }
  return(disReward)
}

discountRewardsAll<- function(currRes,discountFactor=0.99) { ###made for maze problem 
  len <- length(currRes)
  for (i in 1:len) {
    reward <- currRes[[i]]$reward
    goodRewards <- reward[which(reward >= -0.01)]
    disRewards <- discountRewards(goodRewards,discountFactor=discountFactor)
    reward[which(reward >= -0.01)] <- disRewards
    currRes[[i]]$reward <- reward
  }
  return(currRes)
}

expectedRewardsDiscounted <- function(currRes,discountFactor=0.99) { 
  lenRes <- length(currRes)
  for (i in 1:lenRes) {
    reward <- currRes[[i]]$reward
    disReward <- NULL
    seqReward <- 0
    len <- length(reward)
    for (j in len:1) {
      seqReward <- seqReward*discountFactor + reward[j]
      disReward[j] <- seqReward
    }
    currRes[[i]]$discountedReward <- disReward
  }
  return(currRes)
}

oneStepValue <- function(currRes,critic=forest,gamma=0.99) { 
  lenRes <- length(currRes)
  for (i in 1:lenRes) {
    reward <- currRes[[i]]$reward
    input <- currRes[[i]]$input
    value <- predict(critic,data.frame(input))
    disValue <- NULL
    len <- length(reward)
    disValue[len] <- reward[len]
    for (j in (len-1):1) {
      disValue[j] <- reward[j] + gamma*value[j+1]
    }
    currRes[[i]]$value <- disValue
  }
  return(currRes)
}

calculateAdvantageRes <- function(currRes) { ###made for maze problem 
  lenRes <- length(currRes)
  for (i in 1:lenRes) {
    reward <- currRes[[i]]$reward
    len <- length(reward)
    advReward <- NULL
    for (j in len:1) {
      advReward[j] <- reward[j] - mean(reward[len:j])
    }
    currRes[[i]]$advantage <- advReward
  }
  return(currRes)
}

calculateAdvantageCritic <- function(currRes, critic) { ##
  lenRes <- length(currRes)
  for (i in 1:lenRes) {
    reward <- currRes[[i]]$discountedReward
    input <- currRes[[i]]$input
    value <- predict(critic,data.frame(input))
    currRes[[i]]$advantage <- reward - value
  }
  return(currRes)
}

calculateAdvantageCriticCorrect <- function(currRes, critic) { ##
  lenRes <- length(currRes)
  for (i in 1:lenRes) {
    reward <- currRes[[i]]$reward
    input <- currRes[[i]]$input
    value <- predict(critic,data.frame(input))
    valueNextState <- c(value[-1],0)
    currRes[[i]]$advantage <- reward + valueNextState - value 
  }
  return(currRes)
}

fpred <- function(x,fit,bestFitness) {
  fit$predAll <- TRUE
  res = predict(fit, x)
  #infillExpectedImprovement(res$y, res$s, bestFitness) #note: this requires that y (current observations) are in the same environment.
  return(res$y)
}

compareBehaviorWeightedReward <- function(x,ann,inputBeh,outputBeh,reward) {
  if(!is.null(x)) {
    activeNodes <- ann$nodes[ann$nodes[,1]==1,]
    activeNodes[,-c(1:2,seq(3,ncol(activeNodes),2))] <- x
    ann$nodes[ann$nodes[,1]==1,] <- activeNodes
  }
  input <- inputBeh
  output <- queryNetworkSoftmax(ann,input)
  #print(output)
  res <- abs(output - outputBeh)*reward
  #print(mean(res))
  mean(res,na.rm=TRUE)/mean(abs(reward)) ###TODO NA?
}

compareValue <- function(ann,input,expReward) {
  value <- queryNetwork(ann,input)
  res <- (value - expReward)^2
  mean(res,,na.rm=TRUE)
}

computeCrossEntropyLossWeighted <- function(x,ann,inputBeh,outputBeh,reward) {
  if(!is.null(x)) {
    activeNodes <- ann$nodes[ann$nodes[,1]==1,]
    activeNodes[,-c(1:2,seq(3,ncol(activeNodes),2))] <- x
    ann$nodes[ann$nodes[,1]==1,] <- activeNodes
  }
  input <- inputBeh
  output <- queryNetworkSoftmax(ann,input)
  ##handle numerical issues
  logOutput <- log(output)
  logOutput[which(is.infinite(logOutput))] <- 1e-100
  res <- - outputBeh*logOutput
  #print(res)
  res <- res * reward
  #print(mean(res))
  mean(res,na.rm=TRUE) ###TODO NA?
}

getOffspring <- function(anns,numberChilds){
  offspring <- NULL
  for (i in 1:numberChilds) {
    offspring <- c(offspring,lapply(anns,mutateWrapper,seed=123,mutationRate=0.05))
  }
  offspring
}

computeFitnessStates <- function(x,steps,instances, stochasticPolicy=TRUE) {
  sum <- 0
  states= NULL
  for(j in 1:instances) {
    episode_rewards = 0
    #fixed seed for no noise, python seed has to be set additionally to R, cause they are independet
    #set.seed(123)
    #py_set_seed(123, disable_hash_randomization = TRUE)
    obs = env$reset()
    outputActions = NULL
    rewards = NULL
    obs_R[[1]] <- obs
    action =  1L
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
        retValue <- list(epReward=sum/instances,input=states,action=outputActions,reward=rewards)
        env$close()
        return(retValue)
      } else {
        states <- rbind(states,c(as.vector(obs_R[[1]]))) ##icl. Bias # final state not stores (since there is no action or reward)
      }
    }
    sum= sum + episode_rewards
  }
  retValue <- list(epReward=sum/instances,input=states,action=outputActions,reward=rewards)
  return(retValue)
}

optimCrossEntropy<- function(anns,input,output,reward,iterations=1000,noElitist=2,plotFitness=TRUE) {
  fitness=NULL
  len <- length(anns)
  numberChilds <- (len-noElitist)/noElitist
  behFit <- unlist(lapply(anns,function(ann) computeCrossEntropyLossWeighted(x=NULL,ann,input,output,reward)))
  elitist <- anns[order(behFit)][1:noElitist]
  behFitElitists <- behFit[order(behFit)][1:noElitist]
  #par(mfrow = c(1, 2))
  if(plotFitness) {
    print("starting first iteration")
  }
  for(i in 1:iterations) {
    offspring <- getOffspring(elitist,numberChilds)
    behFitOffspring <- unlist(lapply(offspring,function(ann) computeCrossEntropyLossWeighted(x=NULL,ann,input,output,reward)))
    anns <- c(elitist,offspring)
    behFit <- c(behFitElitists,behFitOffspring)
    elitist <- anns[order(behFit)][1:noElitist]
    behFitElitists <- behFit[order(behFit)][1:noElitist]
    if((i==1 | i %% 100 == 0) & plotFitness) {
      fitness <- c(fitness,min(behFit))
      #plot(fitness,main=min(behFit))
      #plot(behFit,main=i)
      print(min(behFit))
    } else if (i==1 | i %% 100 == 0) {
      #print(i)
      #print(min(behFit))
    }
  }
  #par(mfrow = c(1, 1))
  return(anns)
}

optimAbsBehaviour<- function(anns,input,output,reward,iterations=1000,noElitist=2,plotFitness=TRUE) {
  fitness=NULL
  len <- length(anns)
  numberChilds <- (len-noElitist)/noElitist
  behFit <- unlist(lapply(anns,function(ann) compareBehaviorWeightedReward(x=NULL,ann,input,output,reward)))
  elitist <- anns[order(behFit)][1:noElitist]
  behFitElitists <- behFit[order(behFit)][1:noElitist]
  #par(mfrow = c(1, 2))
  for(i in 1:iterations) {
    offspring <- getOffspring(elitist,numberChilds)
    behFitOffspring <- unlist(lapply(offspring,function(ann) compareBehaviorWeightedReward(x=NULL,ann,input,output,reward)))
    anns <- c(elitist,offspring)
    behFit <- c(behFitElitists,behFitOffspring)
    elitist <- anns[order(behFit)][1:noElitist]
    behFitElitists <- behFit[order(behFit)][1:noElitist]
    if((i==1 | i %% 100 == 0) & plotFitness) {
      fitness <- c(fitness,min(behFit))
      plot(fitness,main=min(behFit))
      plot(behFit,main=i)
      print(min(behFit))
    } else if (i==1 | i %% 100 == 0) {
      #print(i)
      #print(min(behFit))
    }
  }
  #par(mfrow = c(1, 1))
  return(anns)
}

optimCritic<- function(anns,input,expReward,iterations=1000,noElitist=2,plotFitness=TRUE) {
  fitness=NULL
  len <- length(anns)
  numberChilds <- (len-noElitist)/noElitist
  criticFitness <- unlist(lapply(anns,function(ann) compareValue(ann,input,expReward)))
  elitist <- anns[order(criticFitness)][1:noElitist]
  criticFitnessElitists <- criticFitness[order(criticFitness)][1:noElitist]
  #par(mfrow = c(1, 2))
  for(i in 1:iterations) {
    offspring <- getOffspring(elitist,numberChilds)
    criticFitnessOffspring <- unlist(lapply(offspring,function(ann) compareValue(ann,input,expReward)))
    anns <- c(elitist,offspring)
    criticFitness <- c(criticFitnessElitists,criticFitnessOffspring)
    elitist <- anns[order(criticFitness)][1:noElitist]
    criticFitnessElitists <- criticFitness[order(criticFitness)][1:noElitist]
    if((i==1 | i %% 100 == 0) & plotFitness) {
      fitness <- c(fitness,min(criticFitness))
      plot(fitness,main=min(criticFitness))
      plot(criticFitness,main=i)
      print(min(criticFitness))
    } else if (i==1 | i %% 100 == 0) {
      #print(i)
      #print(min(behFit))
    }
  }
  #par(mfrow = c(1, 1))
  return(anns)
}


obs_R <- list()

softmaxNavigation_policy <- function(x,obs) {
  inputs <-  as.vector(obs[[1]])
  #inputs <- c(1,inputs) #include bias
  output <- queryNetworkSoftmax(ann=x,input=inputs)
  outputClass <- (which.max(output) -1)
  return(as.integer(outputClass))
}

softmaxNavigation_policy_stochastic <- function(x,obs) {
  inputs <-  as.vector(obs[[1]])
  output <- queryNetworkSoftmax(ann=x,input=inputs)
  outputClass <- as.integer(sample(0:1,1,prob=output))
  return(as.integer(outputClass))
}

#compute fitness, x is an chromosome
computeFitness <- function(x,instances) {
  sum <- 0
  for(j in 1:instances) {
    episode_rewards = 0
    set.seed(123)
    py_set_seed(123, disable_hash_randomization = TRUE)
    obs = env$reset()
    obs_R[[1]] <- obs
    action =  c(1L,1L)
    for (i in 1:1000) {
      action = softmaxNavigation_policy(x, obs_R)
      obs_R= env$step(action)
      episode_rewards= episode_rewards + obs_R[[2]][1]
      if (obs_R[[3]]==TRUE) {
        break
      }}
    sum= sum+episode_rewards
  }
  return(sum/instances)
}

testInstance<- function(x,instances,steps=200,stochastic=FALSE) {
  sum <- 0
  for(i in 1:instances) {
    episode_rewards = 0
    obs = env$reset()
    obs_R[[1]] <- obs
    action =  c(1L,1L)
    for (i in 1:steps) {
      if(stochastic) {
        action = softmaxNavigation_policy_stochastic(x, obs_R)
      } else {
        action = softmaxNavigation_policy(x, obs_R)
      }
      obs_R= env$step(action)
      #Sys.sleep(0.5)
      #env$render()
      episode_rewards= episode_rewards + obs_R[[2]][1] ##normal
      #states <- rbind(states,obs_R[[1]])
      if (obs_R[[3]]==TRUE) {
        break
      }}
    sum= sum+episode_rewards
  }
  #env$render(mode='close')
  env$close()
  return(sum/instances)
}

testInstanceRender<- function(x,instances) {
  sum <- 0
  for(i in 1:instances) {
    episode_rewards = 0
    obs = env$reset()
    obs_R[[1]] <- obs
    action =  c(1L,1L)
    for (i in 1:100) {
      action = softmaxNavigation_policy_stochastic(x, obs_R)
      obs_R= env$step(action)
      Sys.sleep(0.5)
      env$render()
      episode_rewards= episode_rewards + obs_R[[2]][1] ##normal
      #states <- rbind(states,obs_R[[1]])
      if (obs_R[[3]]==TRUE) {
        break
      }}
    sum= sum+episode_rewards
  }
  #env$render(mode='close')
  env$close()
  return(sum/instances)
}

####Main LOOOP starts here

  ##params
  numInputs <- 4 #70input+1Bias
  numOutputs <- 2
  numSamples <- 1
  funcSet <- c("tanh","sig","gauss","soft","step","relu")
  params <- c(numInputs,numNodes=200,numOutputs,nodeArity=10)
  criticParams <- c(numInputs=4,numNodes=200,numOutputs=1,nodeArity=10)
  popSize <- 5
  initSize <- popSize*2
  elitistSelectionSize <- popSize
  elitistArchiveSize <- popSize*2
  
  ###Iterative Variables init
  fitness <- NULL
  allTest <- NULL
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
  currRes <- lapply(anns,computeFitnessStates,steps=200,instances=1,stochasticPolicy=TRUE)
  
  ###Function LOOP
  for (i in 1:50) {
    print(paste("Iteration:",i))
    ####### Testing and generating EXPERIENCE, use stochastic policy for init gen
    if(i>1) {
      currRes <- lapply(anns,computeFitnessStates,steps=200,instances=1,stochasticPolicy=FALSE)
    }
    
    #Testing
    #if(i>1) {
    #  eliteRes <- computeFitnessStates(anns[[1]],steps=200,instances=1,stochasticPolicy=TRUE)
    #  currRes <- lapply(anns[-1],computeFitnessStates,steps=200,instances=1,stochasticPolicy=FALSE)
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
            eliteRes <- computeFitnessStates(eliteFitAnn,steps=200,instances=1,stochasticPolicy=FALSE)
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
            challengerRes <- computeFitnessStates(anns[[challenger[j]]],steps=200,instances=1,stochasticPolicy=FALSE)
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
    currLearnExperience <- extractExperienceAdvantageSimple(eliteExperience, selSize= experienceSelectionSize) ## Selection 20% worksquite good
    ## behavior distance, take % of random states to compute distance
    #currModelExperience <- extractExperienceRandom(eliteExperience, selSize= 0.5) ## TODO Testing
    
    goodExperienceInputs <- currLearnExperience$input
    goodExperienceReward <- currLearnExperience$advantage
    goodExperienceOutputs <- currLearnExperience$output
    
   
    
    if(diskLogging==TRUE)
    {
      ##main plot
      pdf(paste(dirName,"/plot",instanceName,".pdf",sep=""))
      plot(fitness,main=(paste("Total Evaluations:",length(fitness),
                               "Total Repeats:",totalRepeats,
                               " Elitist Fitness(avg):",mean(eliteRepeats),
                               " Elitist Repeated:",length(eliteRepeats))),ylim=c(10,200))
      abline(h=200, col="red")
      abline(v=initSize, col="blue")
      dev.off()
      ##store data global
      #save(fitness,bestMethodCounter,file=paste(dirName,"/result",instanceName,".Rdata",sep=""))
      #save(allResDiscounted,file=paste(dirName,"/data",instanceName,".Rdata",sep=""))
    }
    
    ###debugging
    if(nrow(goodExperienceInputs) != nrow(goodExperienceOutputs)) {
      print("Wrong Number of input rows! experienceInput")
    }
    if(any(is.na(goodExperienceInputs))) {
      print("NA in input, Break")
      break
    }
    if(ncol(goodExperienceInputs) != 4) {
      print("Wrong Number of input colums! experienceInput")
    }
    for (debugAnn in modelRes) {
      if(ncol(debugAnn$input) != 4) {
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
    
    ##new SurrOptim
    print("SMB-NE")
    
    behaviorDistanceKriging <- function(annA,annB,input=currModelExperience$input, reward=currModelExperience$advantage) {
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
    anns <- list(eliteFitAnn,eliteCrossEntropyAnn,eliteAbsBehAnn,bestExpAnn,mutateElite)
    
    ###Watch for Duplicates / 
    dupli= anyDuplicated(anns)
    while (dupli) {
      print("duplicates!!")
      print(duplicated(anns))
      anns[[dupli]] <- mutateWrapper(anns[[dupli]],1337,mutationRate = 0.01)
      dupli= anyDuplicated(anns)
    }
    
    for (id in 1:4) {
      anns[[id+1]]$id <- (initSize+(i-1)*popSize)+id
    }
    print("Testing Stopping Criteria")
    testing <- sapply(anns,testInstance,instances=100)
    print(testing)
    logRes <- c(length(fitness),testing)
    allTest <- rbind(allTest,logRes)
    bestMethodCounter <- c(bestMethodCounter,which.max(testing))
    print("Best Method Counter:")
    print(bestMethodCounter)
    if(any(testing >=195 )) {
      print("Found Solution")
      #write.csv(allTest,file=paste(dirName,"/finalResult",instanceName,".csv",sep=""))
      break
    }
  }
#write.csv(allTest,file=paste(dirName,"/allTestResult",instanceName,".csv",sep=""))
#write.csv(c(length(fitness),fitness),file=paste(dirName,"/finalResult",instanceName,".csv",sep=""))



