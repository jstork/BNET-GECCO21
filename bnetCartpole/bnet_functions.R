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

hush=function(code){
  sink("sinkTest") # use /dev/null in UNIX
  tmp = code
  sink()
  return(tmp)
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

actionsToOutput <- function(experienceAction,numOutputs) {
  output<- rep(0,numOutputs) ####number hardoced!!! TODO
  output[experienceAction+1] <- 1
  return(output)
} 

actionsToAvoid <- function(experienceAction,numOutputs) {
  output<- rep(0,numOutputs) ####number hardoced!!! TODO
  output[experienceAction+1] <- 1
  return(output)
} 

extractExperience <- function(res,numOutputs) {
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
    expPosOutput<- rbind(expPosOutput,actionsToOutput(selPosAction,numOutputs))
    expNegOutput<- rbind(expNegOutput,actionsToAvoid(selNegAction,numOutputs))
  }
  return(list(input=expInput,posOutput=expPosOutput,negOutput=expNegOutput))
}

extractExperienceDiscounted <- function(res,numOutputs) {
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
    expNegOutput<- rbind(expNegOutput,actionsToAvoid(selNegAction,numOutputs))
  }
  return(list(input=expInput,posOutput=expPosOutput,negOutput=expNegOutput,rewardArchive=expRewardArchive))
}

extractExperienceRewardSimple <- function(res,numOutputs,selSize=0.2) {
  input <- res$input[order(res$reward),]
  action <- res$action[order(res$reward)]
  reward <- res$reward[order(res$reward)]
  len <- length(reward)
  sel <- len * selSize
  from <- len - sel
  expInput <- input[from:len,]
  expRewardArchive <- reward[from:len]
  expPosAction <- action[from:len]
  expPosOutput <- t(sapply(expPosAction,actionsToOutput,numOutputs=numOutputs))
  expNegOutput=NULL
  return(list(input=expInput,output=expPosOutput,negOutput=expNegOutput,reward=expRewardArchive))
}

extractExperienceAdvantageSimple <- function(res,numOutputs,selSize= 1) {
  input <- res$input[order(res$advantage),]
  action <- res$action[order(res$advantage)]
  advantage <- res$advantage[order(res$advantage)]
  len <- length(advantage)
  sel <- len * selSize
  from <- len - sel
  expInput <- input[from:len,]
  expAdvantage <- advantage[from:len]
  expPosAction <- action[from:len]
  expPosOutput <- t(sapply(expPosAction,actionsToOutput,numOutputs=numOutputs))
  return(list(input=expInput,output=expPosOutput,advantage=expAdvantage))
}

extractExperienceRandom <- function(res,numOutputs,selSize= 0.2) {
  input <- res$input[order(res$advantage),]
  action <- res$action[order(res$advantage)]
  advantage <- res$advantage[order(res$advantage)]
  len <- length(advantage)
  numSel <- len * selSize
  sel <- sample(1:len,numSel)
  expInput <- input[sel,]
  expAdvantage <- advantage[sel]
  expPosAction <- action[sel]
  expPosOutput <- t(sapply(expPosAction,actionsToOutput,numOutputs=numOutputs))
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

trainCritic <- function(allResDiscounted,iteration) {
  allResInputs <- lapply(allResDiscounted, FUN= '[[',2)
  
  ###### TODO: Fitting Q(s,a) -> not Value! Q: r(s,a) + V(s+1)
  ####  Correct?   Bootstrapped value function 
  if(iteration==1)
  { ###restart R
    critic = keras_model_sequential() %>% 
      layer_dense(units=128, activation="tanh", input_shape=numInputs) %>% 
      layer_dense(units=64, activation = "tanh") %>% 
      layer_dense(units=1, activation="linear")
    
    critic %>% compile(
      loss = "mae",
      optimizer =  "adam" 
      #metrics = list("mean_absolute_error")
    ) 
  }
  
  allValues <- lapply(allResDiscounted, FUN= '[[',5) ## init V(s) to discounted sum R(s)
  
  learnInputs <- do.call(rbind,allResInputs)
  learnValues  <- do.call('c',allValues)
  print("Fitting Critic")
  critic %>% fit(learnInputs, learnValues, epochs = 2000,verbose = 0)
  scores = critic %>% evaluate(learnInputs, learnValues, verbose = 0)
  print(scores)
  return(critic)
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

calculateAdvantageKeras <- function(currRes, model) { ##
  lenRes <- length(currRes)
  for (i in 1:lenRes) {
    reward <- currRes[[i]]$discountedReward
    input <- currRes[[i]]$input
    #print(typeof(input))
    value = model %>% predict(input)
    currRes[[i]]$advantage <- reward - value
  }
  return(currRes)
}

calculateAdvantageKerasCorrect <- function(currRes, model, gammaDis=0.99) { ##
  lenRes <- length(currRes)
  for (i in 1:lenRes) {
    reward <- currRes[[i]]$reward
    input <- currRes[[i]]$input
    #print(typeof(input))
    value = model %>% predict(input)
    valueNextState <- c(value[-1],0)
    currRes[[i]]$advantage <- reward + gammaDis*valueNextState - value 
  }
  return(currRes)
}

fpred <- function(x,fit,bestFitness) {
  fit$predAll <- TRUE
  res = predict(fit, x)
  expImp <- infillExpectedImprovement(res$y, res$s, bestFitness) #note: this requires that y (current observations) are in the same environment.
  return(res$y)
  #print(expImp)
  #return(expImp)
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
  mean(res,na.rm=TRUE)
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
  reward[which(reward < 0)] <- 0 
  res <- res * reward
  #print(mean(res))
  mean(res,na.rm=TRUE) ###TODO NA?
}

getOffspring <- function(anns,numberChilds,mutationRate=0.05){
  offspring <- NULL
  for (i in 1:numberChilds) {
    offspring <- c(offspring,lapply(anns,mutateWrapper,seed=123,mutationRate)) ##was 0.05
  }
  offspring
}

computeFitnessStates <- function(x,steps,instances, stochasticPolicy=TRUE,fitSign=1,stochasticExploration=FALSE,explorationRate = 0.1) {
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
      if(stochasticExploration) {
        action = softmaxNavigation_policy_stochastic_exploration(x, obs_R, explorationRate)
      }
      obs_R= env$step(action)
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
        retValue <- list(epReward=(sum/instances)*fitSign,input=states,action=outputActions,reward=rewards)
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

optimCrossEntropy<- function(anns,input,output,reward,iterations=1000,noElitist=2,numberChilds=20,mutationRate=0.05,plotFitness=TRUE) {
  fitness=NULL
  numberChilds <- numberChilds/noElitist
  behFit <- unlist(lapply(anns,function(ann) computeCrossEntropyLossWeighted(x=NULL,ann,input,output,reward)))
  elitist <- anns[order(behFit)][1:noElitist]
  behFitElitists <- behFit[order(behFit)][1:noElitist]
  #par(mfrow = c(1, 2))
  if(plotFitness) {
    print("starting first iteration")
  }
  for(i in 1:iterations) {
    offspring <- getOffspring(elitist,numberChilds,mutationRate)
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
  return(elitist)
}



optimAbsBehaviour<- function(anns,input,output,reward,iterations=1000,noElitist=2,numberChilds=20,mutationRate=0.05,plotFitness=TRUE) {
  fitness=NULL
  numberChilds <- numberChilds / noElitist
  behFit <- unlist(lapply(anns,function(ann) compareBehaviorWeightedReward(x=NULL,ann,input,output,reward)))
  elitist <- anns[order(behFit)][1:noElitist]
  behFitElitists <- behFit[order(behFit)][1:noElitist]
  #par(mfrow = c(1, 2))
  for(i in 1:iterations) {
    offspring <- getOffspring(elitist,numberChilds,mutationRate)
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
  return(elitist)
}

optimKriging <- function(anns,model,fitnessArchive,iterations=1000,noElitist=2,numberChilds=8,mutationRate=0.01) {
  numberChilds <- numberChilds / noElitist
  predictedMean <- fpred(anns,model,min(-fitnessArchive))
  elitist <- anns[order(predictedMean)][1:noElitist]
  surrFitElitists <- predictedMean[order(predictedMean)][1:noElitist]
  for (i in 1:iterations) {
    offspring <- getOffspring(elitist,numberChilds,mutationRate)
    surrFitOffspring <- fpred(offspring,model,min(-fitnessArchive))
    anns <- c(elitist,offspring)
    surrFit <- c(surrFitElitists,surrFitOffspring)
    elitist <- anns[order(surrFit)][1:noElitist]
    surrFitElitists <- surrFit[order(surrFit)][1:noElitist]
  }
  print(paste("SMBNE Predicted Mean:",surrFitElitists[1]))
  return(elitist)
}

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
  outputClass <- as.integer(sample(0:(x$meta[3]-1),1,prob=output))
  return(as.integer(outputClass))
}

softmaxNavigation_policy_stochastic_exploration <- function(x,obs,explorationRate) {
  inputs <-  as.vector(obs[[1]])
  output <- queryNetworkSoftmax(ann=x,input=inputs)
  outputClass <- as.integer(sample(0:(x$meta[3]-1),1,prob=output))
  probExplore <- output[outputClass+1] * explorationRate #chance for "extra" exploration is less, if unlikley action was chosen
  if (probExplore >= runif(1)) {
    newProb <- 1 - output #more likley to chose new actions
    newProb[outputClass+1] <- 0
    #print(output)
    #print(outputClass)
    #print(newProb)
    outputClass <- as.integer(sample(0:(x$meta[3]-1),1,prob=newProb))
    #print(outputClass)
  }
  return(as.integer(outputClass))
}

#compute fitness, x is an chromosome
computeFitness <- function(x,instances,fitSign=1) {
  sum <- 0
  obs_R <- list()
  for(j in 1:instances) {
    episode_rewards = 0
    set.seed(123)
    py_set_seed(123, disable_hash_randomization = TRUE)
    obs = env$reset()
    obs_R[[1]] <- obs
    action =  rep(1L,x$meta[3])
    for (i in 1:1000) {
      action = softmaxNavigation_policy(x, obs_R)
      obs_R= env$step(action)
      episode_rewards= episode_rewards + obs_R[[2]][1]
      if (obs_R[[3]]==TRUE) {
        break
      }}
    sum= sum+episode_rewards
  }
  return((sum/instances)*fitSign)
}

testInstance<- function(x,instances,steps=200,stochastic=FALSE,fitSign=1) {
  sum <- 0
  obs_R <- list()
  for(i in 1:instances) {
    episode_rewards = 0
    obs = env$reset()
    obs_R[[1]] <- obs
    action =  rep(1L,x$meta[3])
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
  return((sum/instances)*fitSign)
}

testInstanceRender<- function(x,instances) {
  sum <- 0
  obs_R <- list()
  for(i in 1:instances) {
    episode_rewards = 0
    obs = env$reset()
    obs_R[[1]] <- obs
    action =  rep(1L,x$meta[3])
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