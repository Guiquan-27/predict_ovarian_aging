library(data.table)
library(dplyr)
library(naniar)
library(purrr)
library(lubridate)
library(survival)
library(Hmisc)
library(missRanger)
library(plotly)
library(readr)
library(mice)
library(caret)
library(parallel)
setwd("/home/louchen/UKB_meno_pre/imputation")


#### load data ####
data_ori <- fread(
  "/home/louchen/UKB_meno_pre/merge_and_extract/data_files/filter_clean/data_clean_surv_predi20250213.txt" ,
  sep = "\t",
  na.strings = c("", "NA")
)
source("~/UKB_meno_pre/imputation/function/imp_missranger.R")


#### Remove high-missing proteins ####
# Remove proteins with >25% missing values

na_ratio <- miss_var_summary(data_ori)

high_na_vars <- na_ratio %>%
  filter(pct_miss > 25) %>%
  pull(variable)
print("high_na_vars")
data_ori <- data_ori %>% 
  select(-all_of(high_na_vars))


#"glipr1" "npm1"   "pcolce" "cst1" 


#### order ordinal / factor ####

source("/home/louchen/UKB_meno_pre/imputation/function/set_ordinal_variables.R")


data_ori <- set_ordinal_variables(data_ori)


# summary(data_ori$blood_pressure_medication_take_cbde)



#### remove disease case < 1% ####

source("~/UKB_meno_pre/imputation/function/remove_disease_0_01.R")

data_ori <- remove_disease_0_01 (data_ori = data_ori)




#### remove nested ####

nested <- c(
  Cs(
    region_code_10,
    #use hba1c to imupute
    glucose_tans_limit,
    # spontaneous_abortion is nested with pregnancy_loss
    spontaneous_abortion,
    # irregular_menses is nested with length_mens_cyc_category
    irregular_menses,
    primary_ovarian_failure,
    polycystic_ovarian_syndrome,
    adenomyosis
    # gestational_diabetes is nested with diabetes_history
    # gestational_diabetes
  )
)
    

dat_nonest <- data_ori %>% 
  select(-any_of(nested))






#### 80% var missing remove ####

rows1 <- nrow(dat_nonest)

# remove participant rows who still have >80% missing after excluding nested vars
dat_final <- dat_nonest[which(rowMeans(is.na(dat_nonest)) < 0.8), ]

rows2 <- nrow(dat_final)

# check sample sizes after removing vars 
paste("n removed excluding missing >80% of vars:", rows1 - rows2)



#### create nelson aalen estimator ####

# create nelson aalen estimator (cumulative hazard rate). Note: needs event indicator to be numeric and not a factor
# dat_final$hazard <- nelsonaalen(dat_final, # dataset
#                            time_follow, # survival time
#                            meno_status) # event indicator




#### create random sample of discovery and validation ####
### random sampling occurs within each level of the event indicator
### should preserve the overall outcome distribution of the data
### https://www.r-bloggers.com/2016/08/data-splitting/

set.seed(2345) # set the seed to make partition reproducible
train.rows <- createDataPartition(y = dat_final$meno_status, 
                                  p = 0.75, # 50% random subsample
                                  list = FALSE) # do not return data as list

# subset the data
dat_disc <- dat_final[train.rows, ]
dat_rep <- dat_final[-train.rows, ]

# check that proportion of cases is maintained
prop.table(table(dat_disc$meno_status))
prop.table(table(dat_rep$meno_status))
prop.table(table(dat_final$meno_status))

# check sample sizes after removing participants 
paste("discovery set n:", (nrow(dat_disc)))

# check sample sizes after removing participants 
paste("replication set n:", (nrow(dat_rep)))

#### dat_disc imputation ####

imp_missranger (dat_impu = dat_disc,
                prefix = "disc",
                save_folder = "/home/louchen/UKB_meno_pre/imputation/impute_data")


#### dat_rep imputation ####


imp_missranger (dat_impu = dat_rep,
                prefix = "rep",
                save_folder = "/home/louchen/UKB_meno_pre/imputation/impute_data")














