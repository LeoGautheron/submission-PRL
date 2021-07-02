checkImpure <- function(w, y){
  if (length(y) < 2)
    return(FALSE);
  ret <- FALSE
  ytmp <- y[w > 0]
  def <- ytmp[1]
  for (i in 2:length(ytmp))
    if (ytmp[i] != def){
      ret <- TRUE
      break;
    }
  ret
}

rpart2TR <- function(tree, bestresponse){
  nbnode <- nrow(tree$frame);
  frame <- tree$frame;
  splits <- tree$splits
  sortIdx <- order(as.numeric(rownames(frame)), decreasing=TRUE);
  sortSplit <- as.numeric(rownames(frame))[which(frame$var != "<leaf>")];
  sortSplit <- order(sortSplit);
  newnodestack <- list();
	pcInit <- sum(tree$model$label == bestresponse);
	ncInit <- length(tree$model$label) - pcInit;
  pInit <- pcInit / (pcInit + ncInit);
  idsplit <- length(sortSplit);
  kidslist <- list();
  parentslist <- array();
  split <- list();
  isleaf <- array();
  ncount <- ncInit;
  pcount <- pcInit;
  d <- c(1);
  depth <- array();
  parentslist[1] <- NA;
  if (nbnode < 2){
    kidslist <- list();
    nbNode <- 1;
  }
  else{
    for (i in sortIdx){
      d <- floor(log2(as.numeric(rownames(frame)[[i]]))) + 1;
      if (frame$var[[i]] == "<leaf>"){
        pcount[i] <- sum(tree$where==i & tree$model$label == bestresponse);
	      ncount[i] <- sum(tree$where==i & tree$model$label != bestresponse);
        depth[i] <- d;
        split[i] <- NA;
        kidslist[i] <- NA;
        isleaf[i] <- TRUE;
      }
      else{
        if ((length(newnodestack) < 2) || (idsplit < 1))
          stop("Error converting rpart tree to TR tree");
        lnode <- newnodestack[[2]];
        rnode <- newnodestack[[1]];
        pcount[i] <- pc <- pcount[lnode] + pcount[rnode];
        ncount[i] <- nc <- ncount[lnode] + ncount[rnode];
        npc <- pcount[lnode];
        nnc <- ncount[lnode];
        idVar <- which(names(tree$model) == frame$var[[i]])
        tmpname <- as.character(frame[i, "var"]);
        if (attributes(tree$terms)$dataClasses[tmpname] == "numeric"){
          br <- as.numeric(splits[, "index"][[sortSplit[[idsplit]]]])
          split[i] <- list(list(idVar=idVar, name=tmpname, breaks=br, type=0));
        }
        else{
          br <- splits[, "index"][[sortSplit[[idsplit]]]];
          lev <- which(tree$csplit[br,] == 3);
          levVal <- attr(tree, "xlevels")[tmpname][[1]];
          lev <- strsplit(levVal, " ")[lev];
          split[i] <- list(list(idVar=idVar, name=tmpname, breaks=lev,
                                type=1));
        }
        kidslist[i] = list(c(lnode, rnode));
        parentslist[lnode] <- parentslist[rnode] <- i;
        depth[i] <- d;
        isleaf[i] <- FALSE;
        newnodestack <- newnodestack[c(-1, -2)]
        idsplit <- idsplit - 1;
      }
      newnodestack <- c(newnodestack, i);
    }
    if ((length(newnodestack) > 1) || (idsplit > 0))
      stop("Error converting rpart tree to TR");
  }
  ret <- list();
  ret$nbNode <- max(sortIdx);
  ret$root <- 1L;
  ret$nodes <- 1:ret$nbNode;
  ret$parentslist <- parentslist;
  ret$kidslist <- kidslist;
  ret$isleaf <- isleaf;
  ret$bestresponse <- bestresponse;
  ret$pcount <- pcount;
  ret$ncount <- ncount;
  ret$depth <- depth;
  ret$split <- split;
  ret;
}

LRCartFusion <- function(tree, pcInit, ncInit){
  listLeafs <- (1:tree$nbNode)[tree$isleaf];
  if (length(listLeafs) < 2)
    return(list(Lnode=tree$root, Rnode=0))
  betaList <- tree$pcount[listLeafs]
  alphaList <- tree$ncount[listLeafs]
  betaList <- betaList / pcInit;
  alphaList <- alphaList / ncInit;
  crVec <- betaList / alphaList;
  listIndex <- order(-crVec)
  alphaListOrd <- cumsum(alphaList[listIndex])
  betaListOrd <- cumsum(betaList[listIndex])
  entropy <- betaListOrd-alphaListOrd
  Lnode <- listLeafs[listIndex[1:which.max(entropy)]]
  Rnode <- listLeafs[listIndex[
                         as.integer(which.max(entropy) + 1):length(listIndex)]]
  list(Lnode=Lnode, Rnode=Rnode)
}

LRCart <- function(formula, data, bestresponse, criterion, maxdepth=10,
                   minsplit=50){
  evaluation <- function(y, wt, parms){
    idx <- y == bestresponse;
    pc <- sum(wt[idx]) * (1 - parms$pInit);
    nc <- sum(wt[!idx]) * parms$pInit;
    label <- bestresponse;
    miss <- nc;
    if (nc > pc){
      label <- parms$neglab;
      miss <- pc;
    }
    return(list(label=label, deviance=miss))
  }
  split <- function(y, wt, x, parms, continuous, ...){
    n <- length(y)
    pInit <- parms$pInit;
    pvec <- y == bestresponse;
    nvec <- !pvec;
    pcount <- sum(pvec * wt)
    ncount <- sum(nvec * wt)
    if (continuous){
      leftpos <- cumsum(pvec * wt)[-n];
      leftneg <- cumsum(nvec * wt)[-n];
      WERMLess <- 2 - (2 * pInit * leftneg / (pcount + ncount) +
                       2 * (1 - pInit) * (pcount - leftpos) /
                       (pcount + ncount));
      WERMGreat <- 2 - (2 * pInit * (ncount - leftneg) / (pcount + ncount) +
                        2 * (1 - pInit) * leftpos / (pcount + ncount));
      ret <- list(goodness=WERMLess, direction=rep(-1, (n - 1)))
      if (max(WERMLess) < max(WERMGreat))
        ret <- list(goodness=WERMGreat, direction=rep(-1, (n - 1)))
    }
    else{
      ux <- sort(unique(x));
      wtsumP <- tapply(wt * pvec, x, sum);
      wtsumN <- tapply(wt * nvec, x, sum);
      werm <- 2 - (2 * pInit * wtsumN / (pcount + ncount) +
                   2 * (1 - pInit) * (pcount - wtsumP) / (pcount + ncount));
      ord <- order(werm);
      no <- length(ord);
      ret <- list(goodness=werm[ord][-no], direction=ux[ord]);
    }
    return(ret);
  }
  init <- function(y, offset, parms, wt){
    pcInit=sum((y == bestresponse) * wt);
    ncInit = sum((y != bestresponse) * wt);
    ntot = pcInit + ncInit;
    pInit = pcInit / ntot;
    neglab <- y[which(y != bestresponse)][[1]]
    list(y=y, parms=list(pInit=pInit, ntot=ntot, poslab=bestresponse,
                         neglab=neglab),
         numresp=1, numy=1, summary=function(yval, dev, wt, ylevel, digits){
        paste("  mean=", format(signif(yval, digits)),
              ", MSE=" , format(signif(dev / wt, digits)), sep='')})
  }
  if (missing(data))
    data <- environment(formula)
  mf <- match.call(expand.dots=FALSE)
  m <- match(c("formula", "data"), names(mf), 0)
  mf <- mf[c(1, m)]
  mf$drop.unused.levels <- FALSE
  mf[[1]] <- as.name("model.frame")
  mf <- eval(mf, parent.frame())
  y <- model.response(mf)
  pcInit <- y == bestresponse;
  ncInit <- sum(!pcInit);
  pcInit <- sum(pcInit);
  pInit <- pcInit / (pcInit + ncInit);
  if (pInit == 1 || pInit == 0)
    return(NULL);
  if (criterion == "TR") {
    alist <- list(eval=evaluation, split=split, init=init);
    rtree <- rpart(formula, mf, method=alist,
                   control=rpart.control(cp=0, maxdepth=maxdepth,
                                         minsplit=minsplit, maxsurrogate=0,
                                         maxcompete=0, minbucket=1),
                   model=TRUE, y=TRUE, xval=0);
  }
  else if (criterion == "gini") {
    rtree <- rpart(formula, mf, method="class",
	                 parms=list(split='gini'),
	                 control=rpart.control(cp=0, maxdepth=maxdepth,
	                                       minsplit=minsplit, maxsurrogate=0,
	                                       maxcompete=0, minbucket=1),
	                 model=TRUE, y=TRUE, xval=0);
  }
  else if (criterion == "info") {
    rtree <- rpart(formula, mf, method="class",
	                 parms=list(split='information'),
	                 control=rpart.control(cp=0, maxdepth=maxdepth,
	                                       minsplit=minsplit, maxsurrogate=0,
	                                       maxcompete=0, minbucket=1),
	                 model=TRUE, y=TRUE, xval=0);
  }
  tree <- rpart2TR(rtree, bestresponse);
  tree$rpart <- rtree;
  tree$bestresponse <- bestresponse;
  splitNode <- LRCartFusion(tree, pcInit, ncInit);
  tree$Lnode <-splitNode$Lnode;
  tree$Rnode <- splitNode$Rnode;
  fustree <- tree$rpart;
  for (i in splitNode$Lnode)
    fustree$frame[i, "yval"] <- -1;
  for (i in splitNode$Rnode)
    fustree$frame[i, "yval"] <- 1;
  tree$fustree <- fustree;
  return(tree);
}

printTR_LRCart<- function(x, ...){
  object <- x;
  id <- object$root;
  nodestack <- id;
  cat("TreeRank tree\n   id) var <|>= threshold #pos:#neg\n\n");
  while(length(nodestack) > 0){
    id <- nodestack[[1]];
    nodestack <- nodestack[-1];
    s <- "";
    sp <- "root";
    if (id != object$root){
      parent <- object$parentslist[id];
      if (object$split[[parent]]["type"] == 0){
        if (object$kidslist[[parent]][[1]] == id)
          sp <- paste(object$split[[parent]]["name"], "<",
                      format(object$split[[parent]]["breaks"], digits=3))
        else
          sp <- paste(object$split[[parent]]["name"], ">=",
                      format(object$split[[parent]]["breaks"], digits=3));
      }
      else{
        if (object$kidslist[[parent]][[1]] == id)
          sp <- paste(object$split[[parent]]["name"], "!=",
                      object$split[[parent]]["breaks"])
        else
          sp <- paste(object$split[[parent]]["name"], "==",
                      object$split[[parent]]["breaks"])
      }
    }
    s <- paste(cat(rep(' ', 2*object$depth[id])), id, "| ", sp, "  ",
               object$pcount[id], ":", object$ncount[id], sep="")
    if (!(object$isleaf[id]))
      nodestack <- c(object$kidslist[[id]][[1]], object$kidslist[[id]][[2]],
                     nodestack)
    else
      s <- paste(s, "*");
    cat(paste(s, "\n"));
  }
}

predictTR_TreeRank <- function(object, newdata, type="score", ...){
  retid <- rep(object$root, nrow(newdata));
  indextab <- list(rep(1:nrow(newdata)));
  nodestack <- list(object$root);
  while(length(nodestack) > 0){
    id <- nodestack[[1]];
    nodestack <- nodestack[-1];
    if (object$isleaf[id]){
      retid[indextab[[id]]] <- id;
      next;
    }
    if (length(indextab[[id]]) > 0){
	    tmp <- predict(object$LRList[[id]]$fustree, newdata[indextab[[id]],],
	                   type="vector");
      kids <- object$kidslist[[id]];
      indextab[kids[1]] <- list(indextab[[id]][tmp < 0]);
      indextab[kids[2]] <- list(indextab[[id]][tmp > 0]);
      nodestack <- c(nodestack, kids[1], kids[2]);
    }
  }
  if (type == "node")
    return(retid)
  return(object$score[retid]);
}

TreeRank <- function(formula, data, bestresponse, criterion, weights=NULL,
                     minsplit=50, maxdepth=10){
  if (missing(data))
    data <- environment(formula)
  mf  <- match.call(expand.dots=FALSE)
  m <- match(c("formula", "data", "weights"), names(mf), 0)
  mf <- mf[c(1, m)]
  mf$drop.unused.levels <- FALSE
  mf[[1]] <- as.name("model.frame")
  mf <- eval(mf, parent.frame())
  response <- model.response(mf);
  w <- model.extract(mf, "weights");
  if (length(w) == 0L)
    w <- rep.int(1, length(response));
  x <- mf[, colnames(mf) != "(weights)"]
  pcInit <- sum(w[response == bestresponse]);
  ncInit <- sum(w[response != bestresponse]);
  id <- 1;
  nextnode <- 1;
  wtmp <- list(seq(1, length(response)));
  kidslist <- list();
  parentslist <- array();
  LRList <- list();
  isleaf <-array();
  score <- array();
  ncount <- ncInit;
  pcount <- pcInit;
  depth <- c(0);
  nodestack <- list(id);
  nodeorder <- array();
  curscore <- 1;
  while(length(nodestack)>0){
    id <- nodestack[[1]];
    isleaf[id] <- TRUE;
    nodestack <- nodestack[-1];
    tmpdata <- x[wtmp[[id]],];
    tmpweights <- w[wtmp[[id]]];
    tmpresponse <- response[wtmp[[id]]];
    pcount[id] <- sum(tmpweights[tmpresponse == bestresponse]);
    ncount[id] <- sum(tmpweights[tmpresponse != bestresponse]);
    nodeorder[id] <- curscore;
    curscore <- curscore + 1;
    if ((depth[id] >= maxdepth) ||
        ((pcount[id] + ncount[id]) < minsplit) ||
        (!checkImpure(tmpweights, tmpresponse))){
      kidslist[id] <- NA;
      LRList[id] <- NA;
      next;
    }
    lrTree <- LRCart(formula=formula, data=tmpdata, maxdepth=maxdepth,
                     bestresponse=bestresponse, minsplit=minsplit,
                     criterion=criterion);
    lrResponse <- predict(lrTree$fustree, tmpdata, type="vector");
    left <- lrResponse <= 0;
    right <- lrResponse > 0;
    npc <- sum(tmpweights[(tmpresponse == bestresponse) & left]);
    nnc <- sum(tmpweights[(tmpresponse != bestresponse) & left]);
    if ((npc + nnc) == 0 || (pcount[id] + ncount[id] - (npc + nnc)) == 0){
       kidslist[id] <- NA;
       LRList[id] <- NA;
       next;
    }
    LRList[id] <- list(lrTree);
    kidslist[id] <- list(c(nextnode + 1, nextnode + 2));
    parentslist[nextnode + 1] <- parentslist[nextnode + 2] <- id;
    depth[nextnode + 1]<- depth[nextnode + 2] <- depth[id] + 1;
    wtmp[nextnode + 1] <- list(wtmp[[id]][left]);
    wtmp[nextnode + 2] <- list(wtmp[[id]][right]);
    isleaf[id] <- FALSE;
    nodestack <- c(nextnode + 1, nextnode + 2, nodestack);
    nextnode <- nextnode + 2;
  }
  nbNode <- nextnode;
  score <- array(0, nbNode);
  leaf <- which(isleaf);
  leafOrdered <- order(nodeorder);
  leafOrdered <- leafOrdered[leafOrdered %in% leaf];
  nbLeaf <- length(leafOrdered);
  for (i in 0:(nbLeaf - 1))
    score[leafOrdered[i + 1]] <- ((nbLeaf - i) / nbLeaf)
  ret <- list();
  ret$leafOrdered <- leafOrdered;
  ret$nodeorder <- nodeorder;
  ret$nodes <- (1:nbNode)
  ret$root <- 1L;
  ret$parentslist <- parentslist;
  ret$kidslist <- kidslist;
  ret$isleaf <- isleaf;
  ret$bestresponse <- bestresponse;
  ret$pcount <- pcount;
  ret$ncount <- ncount;
  ret$depth <- depth;
  ret$LRList <- LRList;
  ret$nbNode <- nextnode;
  ret$score <- score;
  ret
}

printTR_TreeRank <- function(x, ...){
  object <- x;
  id <- object$root;
  nodestack <- id;
  cat("TreeRank tree\n   id) #pos:#neg score\n\n");
  while(length(nodestack) > 0){
    id <- nodestack[[1]];
    nodestack <-nodestack[-1];
    s <- "";
    s <- paste(cat(rep(' ', 2*object$depth[id])), id, "| ", object$pcount[id],
               ":", object$ncount[id], " ",
               format(object$score[id], digits=3), sep="")
    if (!(object$isleaf[id])){
      s <- paste(s, object$LRList[id][[1]]$Rnode);
      nodestack <- c(object$kidslist[[id]][[1]], object$kidslist[[id]][[2]],
                     nodestack);
    }
    else
      s <- paste(s, "*");
    cat(paste(s, "\n"));
  }
}
