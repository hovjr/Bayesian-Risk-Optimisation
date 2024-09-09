library(MAMS)

MAM <- function(n_inter, alpha, beta, std, interesting, uninteresting, 
                sfu, n_sims, true_mu, true_std, iter_nxy) {
  
  # Interesting effect delta is used for the efficacy boundary and uninteresting 
  # effect delta0 is used for futility?
  mam_temp <- mams(K=1, J=n_inter, alpha=alpha, power=1-beta, p=NULL, p0=NULL,
                   r=1:n_inter, r0=1:n_inter, ushape=sfu, lshape=sfu, 
                   delta=interesting, delta0=uninteresting, sd=std, 
                   nstart=1, nstop=NULL, sample.size=FALSE, N=15, type="normal", 
                   parallel=FALSE, print=FALSE)
  
  # currently not used, iterate over different total sizes in python
  # group_size_m <- mam_temp$N / n_inter
  # n_xy_k <- group_size_m / 2
  
  # true_mu and true_std are the posterior hyperparameters used in designing 
  # phase 3. Using the mean and std will return the expectation, in practice 
  # modifying this can be used for demonstrating unexpected bad data in phase 3.
  sim_temp <- mams.sim(nsim=n_sims, nMat=matrix(c(iter_nxy, iter_nxy), 
                                                nrow=n_inter, ncol=2),
                       l=mam_temp$l, u=mam_temp$u, pv=NULL, 
                       deltav=true_mu, sd=true_std, ptest=c(1))
  
  return(c(sim_temp$exss, sim_temp$power))
}

# MAM(n_inter=2, alpha=0.025, beta=0.1, std=45,
#     interesting=10, uninteresting=0, sfu="obf",
#     n_sims=1000, true_mu=15, true_std=45, iter_nxy=30)

