args = commandArgs(trailingOnly = TRUE)
# njc added 2023/02/10
# options(scipen=999)

library(optparse)
library(ashr)
library(Rmpfr)
library(dplyr)

data_dir = '/n/scratch/users/n/njc12/smiles/data'

get_arguments <- function() {
option_list = list(
    make_option(c("--phen"), action="store", type="character", default=NULL,
                help="phenotype to run", metavar="phenotype"),
    make_option(c("--rds"), action="store", type="character", default=FALSE,
                help="read in existing rds for ash", metavar="rds"),
    make_option(c("--maf"), action="store", type="numeric", default=0.01,
                help="MAF to filter by (keep SNPs with MAF>threshold", metavar="MAF"),
    make_option(c("--betahat"), action="store", type='character', default='beta',
                help='ash betahat argument', metavar='betahat'),
    make_option(c("--mixcompdist"), action="store", type='character', default='normal',
                help='ash mixcompdist argument', metavar='betahat'),
    make_option(c("--pointmass"), action="store", type='numeric', default=TRUE,
                help='whether to include point mass at zero as a component in mixture distribution (1=TRUE, 0=FALSE, default: TRUE)', metavar='')
)
opt_parser = OptionParser(option_list=option_list)
args = parse_args(opt_parser)
return(args)
}

add_n = function(pheno) {
    phenotypes <- data.frame(
        phenotype = c('bmi', 'dbp', 'fvc', 'grip_strength', 'hdl', 'height', 'ldl', 'pulse_rate', 'rbc', 'sbp', 'triglycerides', 'urate', 'wbc', 'bc', 'ibd', 'pc', 'diverticulitis', 'gallstones', 'glaucoma', 'hypothyroidism', 'malignant_neoplasms', 'uterine_fibroids', 'varicose_veins', 'arthrosis', 'asthma', 'ibd_finn'),
        n = c(419163, 417001, 383471, 418827, 367021, 419596, 398402, 316932, 407995, 417001, 400639, 400469, 407990, 228951, 86640, 726838, 365929, 420531, 420531, 420473, 420531, 420531, 389464, 420531, 420531, 335453))
    if (pheno %in% phenotypes$phenotype) {
        return(phenotypes[phenotypes$phenotype == pheno, 'n'])
    } else {
        return(NA)
    }
}

read_ss <- function(fname) {
    print(paste('Reading', fname))
    df = read.csv(file = fname, sep = '\t', colClasses=c('pval'='character'))

    if (!('n' %in% names(df))) {
        df$n = add_n(phen)
    }

    if ('neglog10_pval' %in% names(df)) {
        names(df)[names(df) == 'neglog10_pval'] = 'neglog10p'
    }

    if('pval' %in% names(df) && !('neglog10p' %in% names(df))) {
        df$neglog10p = as.numeric(log10(mpfr(df$pval)))*-1
        df$pval = as.numeric(df$pval)
    }
    
    if (!('raf' %in% colnames(df))) {
        if ('af' %in% colnames(df)) {
        df$raf = ifelse(df$beta > 0, df$af, 1 - df$af)
        df$rbeta = abs(df$beta)
        } else if ('af_controls' %in% colnames(df)) {
        df$raf = ifelse(df$beta > 0, df$af_controls, 1 - df$af_controls)
        df$rbeta = abs(df$beta)
        } else {
        stop(paste('No allele frequency column found in', fname))
        }
    }
    if ('neglog10p' %in% colnames(df) && !('pval' %in% colnames(df))) {
        df$pval = 10**(-1*df$neglog10p)
        # df$pval2 = mpfr(df$neglog10_pval, precBits=128)
        # df$pval2 = 10**(-1*df$pval2)
    }
    return(df)
}

invar = function(df) {
    df = df %>% mutate(UKBB_z = sign(UKBB_beta) * abs(qnorm(UKBB_pval/2)), FINNGEN_z = sign(FINNGEN_beta) * abs(qnorm(FINNGEN_pval/2))) %>% 
        mutate(UKBB_se = UKBB_beta / UKBB_z, FINNGEN_se = FINNGEN_beta / FINNGEN_z) %>%
        mutate(se = 1 / sqrt(1 / UKBB_se**2 + 1 / FINNGEN_se**2)) #%>% 
    return(df)
}

main <- function(phen, betahat, mixcompdist, pointmass, maf) {
    cat(paste('Running ash for', phen,'\n'))

    rdsname = sprintf('%s/ash.full_genome.%s.RDS', data_dir, phen)

    if (!rds) {
        pruned_fname = sprintf('%s/fifth.gwas.%s.b37.tsv.gz', '/n/scratch/users/n/njc12/smiles/data', phen)
        ld.pruned = read_ss(fname=pruned_fname)
        cat(paste('...LD-pruned file', pruned_fname, 'has', nrow(ld.pruned), 'rows...\n'))

        ash_betahat = as.numeric(ld.pruned$beta)
        ash_sebetahat = as.numeric(ld.pruned$se)

        ld.pruned.ash = ash.workhorse(ash_betahat, ash_sebetahat,
                                        mixcompdist = mixcompdist,
                                        pointmass=pointmass)


        saveRDS(ld.pruned.ash, rdsname)
    }
    if (rds) {
        cat(paste('RDS name:', rdsname, '\n'))
        ld.pruned.ash = readRDS(rdsname)
    }
    # njc done
    ld.pruned.g = get_fitted_g(ld.pruned.ash)

    
    finngen_pheno = phen %in% c('arthrosis', 'asthma', 'diverticulitis', 'gallstones', 'glaucoma', 'hypothyroidism', 'malignant_neoplasms', 'uterine_fibroids', 'varicose_veins')
    if (finngen_pheno) {
        # cojo = cojo[, c('SNP', 'chr', 'pos', 'A1', 'A2', 'UKBB_af_alt', 'UKBB_beta', 'UKBB_se', 'pval')]
        cojo_fname = sprintf('~/smiles/finngen/finngen.%s.tsv', phen)
        cojo = read.table(cojo_fname, header=T, as.is=T, sep='\t')
        cat(paste('...Finngen file', cojo_fname, 'has', nrow(cojo), 'rows...\n'))
        cojo$snp = paste(cojo$chrom, cojo$pos, cojo$ref, cojo$alt, sep=':')
        cojo = invar(cojo)
        cojo$PosteriorMean[cojo$se == 0 & cojo$PosteriorMean == 0] = cojo$rbeta[cojo$se == 0 & cojo$PosteriorMean == 0]
        cojo = cojo[, c('snp', 'chrom', 'pos', 'ref', 'alt', 'UKBB_af_alt', 'beta', 'se', 'pval')]
        names(cojo) = c('snp', 'chr', 'pos', 'a1', 'a2', 'af', 'beta', 'se', 'pval')
        cojo$raf = ifelse(cojo$beta > 0, cojo$af, 1 - cojo$af)
        cojo$rbeta = abs(cojo$beta)
        genome.wide = cojo
    } else {
        cat(paste0('...Reading in genome-wide file...\n'))

        genomewide_fname = sprintf('%s/gwas.%s.b37.tsv.gz', '/n/scratch/users/n/njc12/smiles/data', phen)

        genome.wide = read_ss(fname=genomewide_fname)
        cat(paste0('...Genome-wide file ', genomewide_fname, ' has ', nrow(genome.wide), ' rows...\n'))
        genome.wide = genome.wide[genome.wide$pval <= 5e-5, ]
        genome.wide$snp = paste(genome.wide$chr, genome.wide$pos, genome.wide$a1, genome.wide$a2, sep=':')
        cat(paste0('...', nrow(genome.wide), ' of these rows pass a weak significance threshold (5e-5)...\n'))

        cojo_fname <- sprintf('~/smiles/cojo_gwas_hits/cojo.%s.5e-5.out.jma.cojo', phen)
        cojo <- read.table(cojo_fname, header=T, as.is=T)
        cat(paste('...COJO file', cojo_fname, 'has', nrow(cojo), 'rows...\n'))
        cojo = cojo[, c('SNP', 'freq_geno', 'bJ', 'bJ_se', 'pJ', 'LD_r')]
        names(cojo) = c('snp', 'topmed_af', 'beta_cojo', 'se_cojo', 'pval_cojo', 'ld_r')
        # Create cojo2 where snp is split by ':' and a1 and a2 are flipped
        cojo2 = cojo
        cojo2$snp = sapply(strsplit(cojo2$snp, ':'), function(x) paste(x[1], x[2], x[4], x[3], sep=':'))
        cojo2$beta_cojo = -1 * cojo2$beta_cojo
        cojo$rbeta_cojo = abs(cojo$beta_cojo)
        cojo2$rbeta_cojo = abs(cojo2$beta_cojo)

        cojo = rbind(cojo, cojo2)

        # names(genome.wide) <- c('chr', 'pos', 'a1', 'a2', 'orig_beta', 'se_orig', 'orig_var_exp', 'orig_var_exp_se', 'eaf', 'raf', 'rbeta_orig', 'pval_orig', 'gwas_n')
        
        genome.wide = merge(genome.wide, cojo, by='snp', all.x=TRUE)

        genome.wide = genome.wide[genome.wide$pval <= 5e-8, ]
    }

    print(head(genome.wide))

    cat(paste('...Starting ash on genome-wide file...\n'))
    cat(sprintf('block_mhc=%s, mixcompdist=%s, betahat=%s, pointmass=%s\n', block_mhc, mixcompdist, betahat, pointmass))

    genome.wide.ash = NULL

    for (chrom in seq(1,22)) {
        df = genome.wide[which(genome.wide$chr==chrom),]
        cat(paste0('...chr', chrom, ' has ', nrow(df), ' SNPs\n'))
        if (nrow(df) == 0) {next}

        ash_betahat = df$beta
        ash_sebetahat = df$se

        df.ash <- ash(ash_betahat, ash_sebetahat,
                    g=ld.pruned.g,
                    fixg=TRUE,
                    mixcompdist = mixcompdist)

        single.chrom.ash = cbind(df, df.ash$result[,c('PosteriorMean','PosteriorSD')])
        if (is.null(genome.wide.ash)) {
            genome.wide.ash = single.chrom.ash
        } else {
            genome.wide.ash = rbind(genome.wide.ash, single.chrom.ash)
        }
        cat(paste('...chrom',chrom,'complete...\n'))
    }
    cat(paste('All chromosomes complete for',phen,'\n'))



    if (!finngen_pheno) {
        cat('...Checking weird cojo loci now...\n')
        # Changed to pval <= 5e-8 not pval_cojo
        big_cojo_change = genome.wide.ash[!is.na(genome.wide.ash$beta_cojo) & genome.wide.ash$pval <= 5e-8 & (genome.wide.ash$beta_cojo / genome.wide.ash$beta > 1.25 | genome.wide.ash$beta_cojo / genome.wide.ash$beta < 0.75), c('chr', 'pos')]
        cat(paste('...Found', nrow(big_cojo_change), 'weird cojo loci...\n'))

        genome.wide.ash$cojo_locus = FALSE

        for (row in 1:nrow(big_cojo_change)) {
        genome.wide.ash$cojo_locus[genome.wide.ash$chr == big_cojo_change[row, 'chr'] & abs(genome.wide.ash$pos - big_cojo_change[row, 'pos']) < 100000] = TRUE
        }
    }

    fname_out = sprintf('%s/ash.genome_wide.%s.tsv.gz', data_dir, phen)

    write.table(genome.wide.ash, gzfile(fname_out), sep='\t', quote=F, row.names = FALSE)
}

args = get_arguments()

phen=args$phen
rds=as.logical(args$rds)

ld_wind_kb = 500
block_mhc = TRUE
maf=args$maf # set maf=NULL to use data that is not MAF filtered

betahat = args$betahat # Which "betahat" is used for ash. options: beta, abs_beta, var_exp, log_var_exp (default: beta)
mixcompdist = args$mixcompdist #"+uniform" # options: normal, +uniform (if using var_exp instead of beta), halfnormal, halfuniform
pointmass = as.logical(args$pointmass) # if True, include point mass at zero in prior. (set to FALSE if using betahat=log_var_exp)

stopifnot(!(((betahat=='abs_beta')|(betahat=='var_exp'))&(mixcompdist!='+uniform'))) # assert that mixcompdist must be +uniform if fitting on abs(beta) or variance explained

main(phen=phen,
    betahat=betahat,
    mixcompdist=mixcompdist,
    pointmass=pointmass,
    maf=maf)


