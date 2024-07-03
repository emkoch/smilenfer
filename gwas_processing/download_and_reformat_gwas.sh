cd /n/scratch/users/n/njc12/smiles/data

# PAN-UKBB ONLY
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/continuous-50-both_sexes-irnt.tsv.bgz; mv continuous-50-both_sexes-irnt.tsv.bgz pan_ukbb.height.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/continuous-21001-both_sexes-irnt.tsv.bgz; mv continuous-21001-both_sexes-irnt.tsv.bgz pan_ukbb.bmi.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/continuous-LDLC-both_sexes-medadj_irnt.tsv.bgz; mv continuous-LDLC-both_sexes-medadj_irnt.tsv.bgz pan_ukbb.ldl.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/biomarkers-30760-both_sexes-irnt.tsv.bgz; mv biomarkers-30760-both_sexes-irnt.tsv.bgz pan_ukbb.hdl.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/continuous-DBP-both_sexes-combined_irnt.tsv.bgz; mv continuous-DBP-both_sexes-combined_irnt.tsv.bgz pan_ukbb.dbp.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/continuous-SBP-both_sexes-combined_medadj_irnt.tsv.bgz; mv continuous-SBP-both_sexes-combined_medadj_irnt.tsv.bgz pan_ukbb.sbp.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/biomarkers-30880-both_sexes-irnt.tsv.bgz; mv biomarkers-30880-both_sexes-irnt.tsv.bgz pan_ukbb.urate.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/continuous-30010-both_sexes-irnt.tsv.bgz; mv continuous-30010-both_sexes-irnt.tsv.bgz pan_ukbb.rbc.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/continuous-30000-both_sexes-irnt.tsv.bgz; mv continuous-30000-both_sexes-irnt.tsv.bgz pan_ukbb.wbc.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/biomarkers-30870-both_sexes-irnt.tsv.bgz; mv biomarkers-30870-both_sexes-irnt.tsv.bgz pan_ukbb.triglycerides.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/continuous-47-both_sexes-irnt.tsv.bgz; mv continuous-47-both_sexes-irnt.tsv.bgz pan_ukbb.grip_strength.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/continuous-20151-both_sexes-irnt.tsv.bgz; mv continuous-20151-both_sexes-irnt.tsv.bgz pan_ukbb.fvc.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/continuous-102-both_sexes-irnt.tsv.bgz; mv continuous-102-both_sexes-irnt.tsv.bgz pan_ukbb.pulse_rate.b37.tsv.gz

wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/phecode-555-both_sexes.tsv.bgz; mv phecode-555-both_sexes.tsv.bgz gwas.ibd_finn.b37.tsv.gz


# PAN-UKBB + Finngen
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/icd10-J45-both_sexes.tsv.bgz; mv icd10-J45-both_sexes.tsv.bgz pan_ukbb.asthma.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/icd10-H40-both_sexes.tsv.bgz; mv icd10-H40-both_sexes.tsv.bgz pan_ukbb.glaucoma.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/phecode-454-both_sexes.tsv.bgz; mv phecode-454-both_sexes.tsv.bgz pan_ukbb.varicose_veins.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/icd10-K80-both_sexes.tsv.bgz; mv icd10-K80-both_sexes.tsv.bgz pan_ukbb.gallstones.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/phecode-740-both_sexes.tsv.bgz; mv phecode-740-both_sexes.tsv.bgz pan_ukbb.arthrosis.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/icd10-K57-both_sexes.tsv.bgz; mv icd10-K57-both_sexes.tsv.bgz pan_ukbb.diverticulitis.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/categorical-20002-both_sexes-1226.tsv.bgz; mv categorical-20002-both_sexes-1226.tsv.bgz pan_ukbb.hypothyroidism.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/icd10-C44-both_sexes.tsv.bgz; mv icd10-C44-both_sexes.tsv.bgz pan_ukbb.malignant_neoplasms.b37.tsv.gz
wget https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/icd10-D25-both_sexes.tsv.bgz; mv icd10-D25-both_sexes.tsv.bgz pan_ukbb.uterine_fibroids.b37.tsv.gz




wget http://www.cardiogramplusc4d.org/media/cardiogramplusc4d-consortium/data-downloads/UKBB.GWAS1KG.EXOME.CAD.SOFT.META.PublicRelease.300517.txt.gz; mv UKBB.GWAS1KG.EXOME.CAD.SOFT.META.PublicRelease.300517.txt.gz cardiogram_meta.cad.tsv.gz
# wget http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST004001-GCST005000/GCST004988/harmonised/29059683-GCST004988-EFO_0000305-build37.f.tsv.gz;
wget http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST004001-GCST005000/GCST004988/harmonised/29059683-GCST004988-EFO_0000305.h.tsv.gz; mv 29059683-GCST004988-EFO_0000305.h.tsv.gz michailidou_2017.bc.b37.tsv.gz
wget http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST003001-GCST004000/GCST003043/IBD_trans_ethnic_association_summ_stats_b37.txt.gz; mv IBD_trans_ethnic_association_summ_stats_b37.txt.gz liu_2015.ibd.b37.tsv.gz
wget https://figshare.com/ndownloader/files/34517828; mv PGC3_SCZ-wave3.european.autosome.public.v3.vcf.tsv.gz pgc_2022.scz.b37.vcf.gz
wget http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90274001-GCST90275000/GCST90274713/GCST90274713.tsv; mv GCST90274713.tsv wang_2023.pc.tsv; gzip wang_2023.pc.tsv
wget http://cnsgenomics.com/data/t2d/Xue_et_al_T2D_META_Nat_Commun_2018.gz

# From http://www.diagram-consortium.org/downloads.html
unzip Mahajan.NatGen2022.DIAMANTE-EUR.sumstat.zip

awk 'NR == 1 {print "snp\trsid\tchr\tpos\ta1\ta2\taf\tbeta\tse\tpval\tn"; next} NR > 1 {a1=toupper($5); a2=toupper($6); print $1":"$2":"a1":"a2"\t"$4"\t"$1"\t"$2"\t"a1"\t"a2"\t"$7"\t"$8"\t"$9"\t"$10"\t251510"}' \
    DIAMANTE-EUR.sumstat.txt | \
    gzip \
    > gwas.t2d.b37.tsv.gz


for pheno in arthrosis asthma bmi dbp diverticulitis fvc gallstones glaucoma grip_strength hdl height hypothyroidism ldl malignant_neoplasms pulse_rate rbc sbp triglycerides urate uterine_fibroids varicose_veins wbc; do
echo $pheno
zcat pan_ukbb.$pheno.b37.tsv.gz | \
awk 'NR == 1 {for (i=1; i<=NF; i++) {f[$i]=i};
    if ("beta_EUR" in f){} else{exit};
    if ("af_controls_EUR" in f)
        {print "snp\tchr\tpos\ta1\ta2\taf_controls\taf_cases\tbeta\tse\tneglog10_pval";
        caseCon=1} else{print "snp\tchr\tpos\ta1\ta2\taf\tbeta\tse\tneglog10_pval"}; next} \
    $1 < 1 || $1 > 22 {next} \
    length($3) != 1 || length($4) != 1 {next} \
    $1 == 6 && $2 > 27500000 && $2 < 35000000 {next} \
    $(f["beta_EUR"]) == "NA" ||  $43 == "true" {next} \
    {if (caseCon != 1)
        {print $1":"$2":"$3":"$4"\t"$1"\t"$2"\t"$3"\t"$4"\t" $(f["af_EUR"]) "\t" $(f["beta_EUR"]) "\t" $(f["se_EUR"]) "\t" $(f["neglog10_pval_EUR"])} \
    if (caseCon == 1)
        {print $1":"$2":"$3":"$4"\t"$1"\t"$2"\t"$3"\t"$4 "\t" $(f["af_controls_EUR"]) "\t" $(f["af_cases_EUR"]) "\t" $(f["beta_EUR"]) "\t" $(f["se_EUR"]) "\t" $(f["neglog10_pval_EUR"])}}' | \
gzip > gwas.$pheno.b37.tsv.gz
done

zcat cardiogram_meta.cad.tsv.gz | \
awk 'BEGIN {print "snp\trsid\tchr\tpos\ta1\ta2\taf\tbeta\tse\tpval\tn"} \
    $3 < 1 ||  $3 > 22 {next} \
    length($5) != 1 || length($6) != 1 {next} \
    $3 == 6 && $4 > 27500000 && $4 < 35000000 {next} \
    {gsub(/_/, ":", $1); print $1"\t"$2 "\t"$3 "\t"$4 "\t"$5 "\t"$6 "\t"$7 "\t"$8 "\t"$9 "\t"$10 "\t"$11}' | \
gzip > gwas.cad.b37.tsv.gz

zcat pgc_2022.scz.b37.vcf.gz | \
awk 'BEGIN {print "snp\trsid\tchr\tpos\ta1\ta2\taf_controls\taf_cases\tbeta\tse\tpval\tn\tneff"} \
    $1 == 6 && $3 > 27500000 && $3 < 35000000 {next} \
    length($4) != 1 || length($5) != 1 {next} \
    $1 > 0 && $1 < 23 {print $1":"$3":"$4":"$5 "\t" $2 "\t" $1 "\t" $3 "\t" $4 "\t" $5 "\t" $7 "\t" $6 "\t" $9 "\t" $10 "\t" $11  "\t" $12+$13 "\t" $14}' | \
gzip > gwas.scz.b37.tsv.gz

zcat wang_2023.pc.tsv.gz | \
awk 'BEGIN {print "snp\tchr\tpos\ta1\ta2\taf\tbeta\tse\tpval"} \
    $1 == 6 && $2 > 27500000 && $2 < 35000000 {next} \
    length($3) != 1 || length($4) != 1 {next} \
    NR > 1 {print $1":"$2":"$3":"$4 "\t" $1 "\t" $2 "\t" $3 "\t" $4 "\t" $7 "\t" $5 "\t" $6 "\t" $8}' | \
gzip > gwas.pc.tsv.gz

zcat liu_2015.ibd.b37.tsv.gz | \
awk 'BEGIN  {print "snp\trsid\tchr\tpos\ta1\ta2\taf\tbeta\tse\tpval"} \
    $2 == 6 && $3 > 27500000 && $3 < 35000000 {next} \
    length($4) != 1 || length($5) != 1 {next} \
    $2 > 0 && $2 < 23 {print $2":"$3":"$4":"$5 "\t" $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" $22 "\t" $10 "\t" $11 "\t" $12}' | \
gzip > gwas.ibd.b37.tsv.gz

#zcat michailidou_2017.bc.b37.tsv.gz | \
#awk 'BEGIN {print "snp\tchr\tpos\ta1\ta2\taf\tbeta\tse\tpval"} \
#    $2 < 1 || $2 > 22 {next} \
#    $2 == 6 && $3 > 27500000 && $3 < 35000000 {next} \
#    length($4) != 1 && length($5) != 1 {next} \
#    {print $2":"$3":"$5":"$4 "\t" $2 "\t" $3 "\t" $5 "\t" $4 "\t" $6 "\t" $7 "\t" $8 "\t" $9}' | \
#gzip > gwas.bc.b37.tsv.gz

zcat michailidou_2017.bc.b37.tsv.gz | \
awk 'BEGIN {print "snp\tchr\tpos\ta1\ta2\taf\tbeta\tse\tpval\tn"} \
    $14 < 1 || $14 > 22 {next} \
    $14 == 6 && $15 > 27500000 && $15 < 35000000 {next} \
    length($16) != 1 || length($17) != 1 {next} \
    {print $14":"$15":"$16":"$17 "\t" $14 "\t" $15 "\t" $16 "\t" $17 "\t" $18 "\t" $19 "\t" $20 "\t" $21 "\t228951"}' | \
gzip > gwas.bc.b37.tsv.gz


zcat Xue_et_al_T2D_META_Nat_Commun_2018.gz | \
awk 'BEGIN {print "snp\trsid\tchr\tpos\ta1\ta2\taf\tbeta\tse\tpval\tn"} \
    $1 < 1 || $1 > 22 {next} \
    $1 == 6 && $2 > 27500000 && $2 < 35000000 {next} \
    length($4) == 1 && length($5) == 1 {print $1":"$2":"$4":"$5 "\t" $3 "\t" $1 "\t" $2 "\t" $4 "\t" $5 "\t" $6 "\t" $7 "\t" $8 "\t" $9 "\t" $10}' | \
gzip > gwas.t2d.tsv.gz
