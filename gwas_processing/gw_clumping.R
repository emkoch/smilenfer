args = commandArgs(trailingOnly = TRUE)

library(optparse)

data_dir = '/n/scratch/users/n/njc12/smiles/data'

get_arguments <- function() {
    option_list = list(
      make_option(c("--phen"), action="store", type="character", default=NULL,
                  help="phenotype to run", metavar="phenotype"),
      make_option(c("--maf"), action="store", type="numeric", default=0.01,
                  help="MAF to filter by (keep SNPs with MAF>threshold", metavar="MAF"),
      make_option(c("--sig"), action="store", type="numeric", default=5e-8,
                  help="Significance threshold", metavar="Sig")
    
    )
    opt_parser = OptionParser(option_list=option_list)
    args = parse_args(opt_parser)
    return(args)
}

args = get_arguments()

phen=args$phen
maf=args$maf
sig=args$sig

file = sprintf('%s/ash.genome_wide.%s.tsv.gz', data_dir, phen)
ash = read.table(file, header=T, sep='\t')

if('neglog10_pval' %in% names(ash) || 'neglog10' %in% names(ash)) {
    names(ash)[names(ash) %in% c('neglog10_pval', 'neglog10')] = 'neglog10p'
}

# finngen_pheno = phen %in% c('arthrosis', 'asthma', 'diverticulitis', 'gallstones', 'glaucoma', 'hypothyroidism', 'malignant_neoplasms', 'uterine_fibroids', 'varicose_veins')

if (!'neglog10p' %in% names(ash) && 'pval' %in% names(ash)) {
    ash$neglog10p = -1*log10(ash$pval)
} else if (!'neglog10p' %in% names(ash)) {
    stop('No p-value column found')
}

if(any(ash$neglog10p < 0)) {
    stop('neglog10p < 0')
}

# Keep only 0.01 < raf < 0.99 and neglog10p > -1*log10(5e-8)
ash = ash[ash$raf > maf & ash$raf < 1-maf & ash$neglog10p > -1*log10(5e-8),]

# Sort decreasing by neglog10p
# ash = ash[order(ash$neglog10p, decreasing=TRUE),]

out = NULL
counter = 0

ash = ash[!(ash$chr == 6 & ash$pos > 27500000 & ash$pos < 35000000), ]

for (chr in 1:22) {
    print(paste('Chr', chr))
    tmp_ash = ash[ash$chr == chr & !is.na(ash$pos) & !is.na(ash$neglog10p),]
    tmp_ash = tmp_ash[order(tmp_ash$neglog10p, decreasing=TRUE),]
    while (nrow(tmp_ash) > 0) {
        # counter = counter + 1
        top = tmp_ash[1,]
        out = rbind(out, top)
        # if (counter %% 1000 == 0) {
        #     print(paste('Counter:', counter))
        #     print(paste('Top:', top$neglog10p))
        # }
        # counter = counter + 1
        
        # tmp_ash = tmp_ash[tmp_ash$chr != top$chr | abs(tmp_ash$pos - top$pos) > 100000,]
        tmp_ash = tmp_ash[abs(tmp_ash$pos - top$pos) > 100000,]
    }
}

write.table(out, sprintf('%s/clumped.genome_wide_ash.%s.tsv', data_dir, phen), sep='\t', quote=FALSE, row.names=FALSE, col.names=TRUE)
