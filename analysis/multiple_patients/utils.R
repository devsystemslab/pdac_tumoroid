read_mdata <- function(file) {
  mu <- reticulate::import("muon")
  mdata <- reticulate::py_to_r(mu$read_h5mu(file))
  return(mdata)
}


pairwise.mahalanobis <- function(x, grouping = NULL, cov = NULL, inverted = FALSE, digits = 5, ...) {
  # standardize input data as matrix
  x <- if (is.vector(x)) {
    matrix(x, ncol = length(x))
  } else {
    as.matrix(x)
  }

  if (!is.matrix(x)) {
    stop("x could not be forced into a matrix")
  }
  # no group assigned, uses first col
  if (length(grouping) == 0) {
    grouping <- t(x[1])
    x <- x[2:dim(x)[2]]
    cat("assigning grouping\n")
    print(grouping)
  }
  # get dims
  n <- nrow(x)
  p <- ncol(x)

  # grouping and matrix do not correspond
  if (n != length(grouping)) {
    cat(paste("n: ", n, "and groups: ", length(grouping), "\n"))
    stop("nrow(x) and length(grouping) are different")
  }
  # groups
  g <- as.factor(grouping)
  # elements in each group
  lev <- lev1 <- levels(g)
  counts <- as.vector(table(g))

  # remove grouping if not represented in data
  if (any(counts == 0)) {
    empty <- lev[counts == 0]
    warning(
      sprintf(ngettext(
        length(empty), "group %s is empty",
        "groups %s are empty"
      ), paste(empty, collapse = " ")),
      domain = NA
    )
    lev1 <- lev[counts > 0]
    g <- factor(g, levels = lev1)
    counts <- as.vector(table(g))
  }

  ng <- length(lev1)
  # g x p matrix of group means from x
  group_means <- tapply(x, list(rep(g, p), col(x)), mean)

  # create covariance matrix, standardize into correlation mtx
  if (missing(cov)) {
    inverted <- FALSE
    cov <- cor(x)
  } else {
    # check cov of correct dimension
    if (dim(cov) != c(p, p)) {
      stop("cov matrix not of dim = (p,p)\n")
    }
  }

  # initialize distance matrix
  distance <- matrix(nrow = ng, ncol = ng)
  dimnames(distance) <- list(rownames(group_means), rownames(group_means))

  means <- round(group_means, digits)
  cov <- round(cov, digits)
  distance <- round(distance, digits)

  for (i in 1:ng) {
    distance[i, ] <- mahalanobis(group_means, group_means[i, ], cov, inverted)
  }

  result <- list(means = group_means, cov = cov, distance = distance, counts = counts)
  return(result)
}
