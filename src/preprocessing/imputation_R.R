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

data_ori <- fread(
  "/home/louchen/UKB_meno_pre/merge_and_extract/data_files/filter_clean/data_clean_surv_predi20250213.txt" ,
  sep = "\t",
  na.strings = c("", "NA")
)




#### miss protein ####
# data <- data_ori
na_ratio <- miss_var_summary(data_ori)

high_na_vars <- na_ratio %>%
  filter(pct_miss > 25) %>%
  pull(variable)

#"glipr1" "npm1"   "pcolce" "cst1" 

data_ori <- data_ori %>% 
  select(-all_of(high_na_vars))

#### order ordinal / factor ####

set_ordinal_variables <- function(data) {
  
  
  
  
  # smoke_status - 设置为有序因子
  data$smoke_status <- factor(
    data$smoke_status,
    ordered = TRUE,
    levels = c("Never", "Previous", "Current")
  )
  
  # bmi_index 是连续变量，保持为数值型
  
  # age_first_birth_category - 设置为有序因子
  data$age_first_birth_category <- factor(
    data$age_first_birth_category,
    ordered = TRUE,
    levels = c("No_live_birth", "Only_1_live_birth" ,"Age<=21", "Age22-25", "Age26-29", "Age>=30" )
  )
  
  # age_last_birth_category - 设置为有序因子
  data$age_last_birth_category <- factor(
    data$age_last_birth_category,
    ordered = TRUE,
    levels = c("No_live_birth",  "Only_1_live_birth","Age<=26", "Age27-30", "Age31-34", "Age>=35")
  )
  
  
  # length_mens_cyc_category - 设置为有序因子
  data$length_mens_cyc_category <- factor(
    data$length_mens_cyc_category,
    ordered = TRUE,
    levels = c(
      "Menopause", "Not_sure_hysterectomy", "Not_sure_other_reason", 
      "Irregular_cycle", "Day<=21", "Day22-27", "Day27-28", "Day>=29"
    )
  )
  
  # oestradiol_category - 设置为有序因子
  data$oestradiol_category <- factor(
    data$oestradiol_category,
    ordered = TRUE,
    levels = c(
      "Lower_limit", "Oestradiol<=Q1", "Oestradiol>Q1<=Q2", 
      "Oestradiol>Q2<=Q3", "Oestradiol>Q3", "Higher_limit"
    )
  )
  
  # shbg_tans_limit 是连续变量，保持为数值型
  
  # total_t_category - 设置为有序因子
  data$total_t_category <- factor(
    data$total_t_category,
    ordered = TRUE,
    levels = c(
      "Lower_limit", "Total_T<=Q1", "Total_T>Q1<=Q2", 
      "Total_T>Q2<=Q3", "Total_T>Q3"
    )
  )
  
  
  
  data$ethnic_bg <- factor(data$ethnic_bg, ordered = FALSE)
  data$ethnic_bg <- relevel(data$ethnic_bg, ref = "White")
  
  data$region_code_10 <- factor(data$region_code_10, ordered = FALSE)
  data$region_code_10 <- relevel(data$region_code_10, ref = "Yorkshire_and_Humber")
  
  data$pregnancy_loss <- factor(data$pregnancy_loss, ordered = FALSE)
  data$pregnancy_loss <- relevel(data$pregnancy_loss, ref = "No")
  
  data$ever_hrt <- factor(data$ever_hrt, ordered = FALSE)
  data$ever_hrt <- relevel(data$ever_hrt, ref = "No")
  
  data$blood_pressure_medication_take_cbde <- factor(data$blood_pressure_medication_take_cbde, ordered = FALSE)
  data$blood_pressure_medication_take_cbde <- relevel(data$blood_pressure_medication_take_cbde, ref = "No")
  
  data$radiotherapy_chemotherapy <- factor(data$radiotherapy_chemotherapy, ordered = FALSE)
  data$radiotherapy_chemotherapy <- relevel(data$radiotherapy_chemotherapy, ref = "No")
  
  data$postpartum_depression <- factor(data$postpartum_depression, ordered = FALSE)
  data$postpartum_depression <- relevel(data$postpartum_depression, ref = "No")
  
  data$prolonged_pregnancy <- factor(data$prolonged_pregnancy, ordered = FALSE)
  data$prolonged_pregnancy <- relevel(data$prolonged_pregnancy, ref = "No")
  
  data$preterm_labour_and_delivery <- factor(data$preterm_labour_and_delivery, ordered = FALSE)
  data$preterm_labour_and_delivery <- relevel(data$preterm_labour_and_delivery, ref = "No")
  
  data$endometriosis <- factor(data$endometriosis, ordered = FALSE)
  data$endometriosis <- relevel(data$endometriosis, ref = "No")
  
  data$gh_pe <- factor(data$gh_pe, ordered = FALSE)
  data$gh_pe <- relevel(data$gh_pe, ref = "No")
  
  data$gestational_diabetes <- factor(data$gestational_diabetes, ordered = FALSE)
  data$gestational_diabetes <- relevel(data$gestational_diabetes, ref = "No")
  
  data$female_infertility <- factor(data$female_infertility, ordered = FALSE)
  data$female_infertility <- relevel(data$female_infertility, ref = "No")
  
  data$ectopic_pregnancy <- factor(data$ectopic_pregnancy, ordered = FALSE)
  data$ectopic_pregnancy <- relevel(data$ectopic_pregnancy, ref = "No")
  
  data$primary_ovarian_failure <- factor(data$primary_ovarian_failure, ordered = FALSE)
  data$primary_ovarian_failure <- relevel(data$primary_ovarian_failure, ref = "No")
  
  data$ovarian_dysfunction <- factor(data$ovarian_dysfunction, ordered = FALSE)
  data$ovarian_dysfunction <- relevel(data$ovarian_dysfunction, ref = "No")
  
  data$leiomyoma_of_uterus <- factor(data$leiomyoma_of_uterus, ordered = FALSE)
  data$leiomyoma_of_uterus <- relevel(data$leiomyoma_of_uterus, ref = "No")
  
  data$excessive_vomiting_in_pregnancy <- factor(data$excessive_vomiting_in_pregnancy, ordered = FALSE)
  data$excessive_vomiting_in_pregnancy <- relevel(data$excessive_vomiting_in_pregnancy, ref = "No")
  
  data$spontaneous_abortion <- factor(data$spontaneous_abortion, ordered = FALSE)
  data$spontaneous_abortion <- relevel(data$spontaneous_abortion, ref = "No")
  
  data$habitual_aborter <- factor(data$habitual_aborter, ordered = FALSE)
  data$habitual_aborter <- relevel(data$habitual_aborter, ref = "No")
  
  data$eclampsia <- factor(data$eclampsia, ordered = FALSE)
  data$eclampsia <- relevel(data$eclampsia, ref = "No")
  
  data$adenomyosis <- factor(data$adenomyosis, ordered = FALSE)
  data$adenomyosis <- relevel(data$adenomyosis, ref = "No")
  
  data$menorrhagia <- factor(data$menorrhagia, ordered = FALSE)
  data$menorrhagia <- relevel(data$menorrhagia, ref = "No")
  
  data$irregular_menses <- factor(data$irregular_menses, ordered = FALSE)
  data$irregular_menses <- relevel(data$irregular_menses, ref = "No")
  
  data$polycystic_ovarian_syndrome <- factor(data$polycystic_ovarian_syndrome, ordered = FALSE)
  data$polycystic_ovarian_syndrome <- relevel(data$polycystic_ovarian_syndrome, ref = "No")
  
  data$cvd_prevalent <- factor(data$cvd_prevalent, ordered = FALSE)
  data$cvd_prevalent <- relevel(data$cvd_prevalent, ref = "No")
  
  data$diabetes_history <- factor(data$diabetes_history, ordered = FALSE)
  data$diabetes_history <- relevel(data$diabetes_history, ref = "No")
  
  
  
  return(data)
}


data_ori <- set_ordinal_variables(data_ori)


# summary(data_ori$blood_pressure_medication_take_cbde)



#### remove disease case < 20 ####


disease_vars <- c(
  Cs(
    radiotherapy_chemotherapy,
    postpartum_depression,
    prolonged_pregnancy,
    preterm_labour_and_delivery,
    endometriosis,
    gh_pe,
    gestational_diabetes,
    female_infertility,
    ectopic_pregnancy,
    primary_ovarian_failure,
    ovarian_dysfunction,
    leiomyoma_of_uterus,
    excessive_vomiting_in_pregnancy,
    spontaneous_abortion,
    habitual_aborter,
    eclampsia,
    adenomyosis,
    menorrhagia,
    irregular_menses,
    polycystic_ovarian_syndrome,
    cvd_prevalent,
    diabetes_history
  )
)


# 创建一个向量，用于存储回答"Yes"的数目少于20的疾病变量名
cat("正在检查疾病变量中'Yes'回答数量...\n")
low_count_disease_vars <- c()

# 遍历每个疾病变量
for (var in disease_vars) {
  if (var %in% colnames(data_ori)) {
    # 计算回答"Yes"的数目
    yes_count <- sum(data_ori[[var]] == "Yes", na.rm = TRUE)
    
    # 如果回答"Yes"的数目少于20，则记录该疾病变量
    if (yes_count < 20) {
      cat(sprintf("疾病变量 '%s' 的'Yes'回答数为 %d，少于阈值20\n", var, yes_count))
      low_count_disease_vars <- c(low_count_disease_vars, var)
    } else {
      cat(sprintf("疾病变量 '%s' 的'Yes'回答数为 %d\n", var, yes_count))
    }
  } else {
    cat(sprintf("警告: 疾病变量 '%s' 不在数据集中\n", var))
  }
}

# 打印需要排除的疾病变量
if (length(low_count_disease_vars) > 0) {
  cat("以下疾病变量将被排除（'Yes'回答数少于20）:\n")
  for (var in low_count_disease_vars) {
    cat(sprintf("- %s\n", var))
  }
  
  # 从data_ori中排除这些疾病变量
  if (length(low_count_disease_vars) > 0) {
    data_ori <- data_ori %>% select(-all_of(low_count_disease_vars))
    cat(sprintf(
      "已从data_ori中排除%d个疾病变量\n",
      length(low_count_disease_vars)
    ))
  }
} else {
  cat("所有疾病变量的'Yes'回答数均达到或超过阈值20，无需排除任何变量\n")
}

# 更新后的变量数
cat(sprintf("data_ori中剩余变量数: %d\n", ncol(data_ori)))




#### remove nested ####

nested <- c(
  Cs(
    region_code_10,
    #use hba1c to imupute
    glucose_tans_limit,
    # spontaneous_abortion is nested with pregnancy_loss
    spontaneous_abortion,
    # irregular_menses is nested with length_mens_cyc_category
    irregular_menses
    # gestational_diabetes is nested with diabetes_history
    # gestational_diabetes
  )
)
    

dat_nonest <- data_ori %>% 
  select(-all_of(nested))






#### 80% var missing remove ####

rows1 <- nrow(dat_nonest)

# remove participant rows who still have >80% missing after excluding nested vars
dat_final <- dat_nonest[which(rowMeans(is.na(dat_nonest)) < 0.8), ]

rows2 <- nrow(dat_final)

# check sample sizes after removing vars 
paste("n removed excluding missing >80% of vars:", rows1 - rows2)

#### create nelson aalen estimator ####

# create nelson aalen estimator (cumulative hazard rate). Note: needs event indicator to be numeric and not a factor
dat_final$hazard <- nelsonaalen(dat_final, # dataset
                           time_follow, # survival time
                           meno_status) # event indicator






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


exp <- fread("/home/louchen/UKB_meno_pre/merge_and_extract/data_files/variable_define/exp_var_v1.txt")
out <- fread("/home/louchen/UKB_meno_pre/merge_and_extract/data_files/variable_define/out_surv_predi_v1.txt")
pro <- fread("/home/louchen/UKB_meno_pre/merge_and_extract/data_files/variable_define/pro_var_v0.txt")

##### clinical var #####

dat_disc_exp <- dat_disc %>%
  select(any_of(c("hazard", exp$Variable, out$Variable)))  


#exclude  some nested variables 

exclude_vars <- c(
  Cs(
    eid,
    time_follow
  )
)
exclude_pro <- c(
  Cs(
    
    fsh_ori,
    lh_ori,
    prl_ori
    
  )
)


dat_disc_exp <- dat_disc_exp %>% 
  select(-all_of(exclude_pro))


exclude_vars <- paste0(exclude_vars, collapse = " - ")
formula <- as.formula(paste0(". ~ . - ", exclude_vars))


# create case weights to weight down the contribution of rows with many missings during the imputation
non_miss_disc_exp <- rowSums(!is.na(dat_disc_exp))



disc_exp <- mclapply(3456:3460, function(x) {
  
  result <- missRanger(
    dat_disc_exp,
    formula = formula,
    maxiter = 10,
    pmm.k = 3,
    verbose = 1,
    seed = x,
    num.trees = 200,
    returnOOB = TRUE,
    case.weights = non_miss_disc_exp
  )
  
  oob <- as.data.frame(attr(result, "oob"))
  colnames(oob) <- c(paste0("Imputation_", x))
  oob$Exposure <- rownames(oob)
  
  seed_num<-paste0("Imputation_",x)
  
  return(list(
    seed = seed_num,
    result = result,
    oob = oob
  ))
  
  
}, mc.cores = 20)



results_dt  <- rbindlist(lapply(disc_exp, function(res) {
  data.table(
    seed = res$seed,
    result = list(res$result),
    oob = list(res$oob)
  )
}))



final_data_list <- results_dt$result
names(final_data_list) <- results_dt$seed


final_oob_list <- results_dt$oob
names(final_oob_list) <- results_dt$seed

merged_oob_all <- Reduce(function(x, y) {
  merge(x, y, by = c("Exposure"), all = TRUE)
}, final_oob_list)

list_disc_exp <- final_data_list
merged_oob_disc_exp <- merged_oob_all


##### protein #####

dat_disc_pro <- dat_disc %>%
  select(any_of(c("age_years",pro$Variable)))  

# import protein name with panel
pro_var_panel_distinct <- fread("/home/louchen/UKB_meno_pre/imputation/pro_var_panel_distinct.csv")

# get all Protein Panel
panels <- unique(pro_var_panel_distinct$Protein_panel)
print(panels)  # 8 different panel

# define seed 
seeds <- 3456:3460  # 5 seed for five different imputation dataset

# create panel-seed combination
tasks <- expand.grid(panel = panels,
                     seed = seeds,
                     stringsAsFactors = FALSE)


run_imputation <- function(task) {
  # get task panel and seed for each group intasks 
  current_panel <-   task$panel
  current_seed <-   task$seed
  
  # Extract the variables corresponding to the current panel
  panel_vars <- pro_var_panel_distinct[Protein_panel == current_panel, Variable]
  cols_to_keep <- c("eid","age_years", panel_vars)
  
  # Create a data subset (automatically skip nonexistent columns)
  dat_panel <- dat_disc_pro %>%
    select(any_of(cols_to_keep))
  
  #test code
  #dat_panel<-dat_panel[,1:4]
  
  
  # Generate formula (exclude EID), with only proteins
  formula <- as.formula(". ~ . - eid")
  
  # Calculate row missing weight
  non_miss <- rowSums(!is.na(dat_panel))
  
  # missranger
  result <- missRanger(
    dat_panel,
    formula = formula,
    maxiter = 10,
    pmm.k = 3,
    verbose = 1,
    seed = current_seed,
    num.trees = 200,
    returnOOB = TRUE,
    case.weights = non_miss
  )
  #Get the corresponding OOB
  oob <- as.data.frame(attr(result, "oob"))
  colnames(oob) <- c(paste0("Imputation_", current_seed))
  oob$Exposure <- rownames(oob)
  oob$Panel<- current_panel
  # Return results (panel name, seed, imputation data)
  result$age_years <- NULL
  
  seed_num<-paste0("Imputation_",current_seed)
  
  return(list(
    panel = current_panel,
    seed = seed_num,
    result = result,
    oob = oob
  ))
}



# Convert to list format for mclapply processing
tasks_list <- split(tasks, seq(nrow(tasks)))

# Perform parallel tasks
results <- mclapply(tasks_list, function(task) {
  run_imputation(task)
}, mc.cores = 20)

# rbind the lists from results(40 lists with panel-seed combination)
results_dt <- rbindlist(lapply(results, function(res) {
  data.table(
    panel = res$panel,
    seed = res$seed,
    result = list(res$result),
    oob = list(res$oob)
  )
}))




#merge the result in the result_dt,result column are List of a series of results
final_merged <- results_dt[, {
  if (.N == 8) {  # Ensure that each seed has 8 panels
    # Sort by panel order to ensure consistency of consolidation
    ordered_results <- result[order(match(panel, panels))]
    # Merge all data frames
    merged <- Reduce(function(x, y) full_join(x, y, by = "eid"), ordered_results)
    list(merged_data = list(merged))
  }
}, by = seed]

final_data_list <- final_merged$merged_data
names(final_data_list) <- final_merged$seed

oob_merged <- results_dt[, {
  # Get the OOB data of all panels under the current seed
  current_oobs <- oob
  # Use rbind to vertically merge all OOB data frames because they have the same column names (impression_[seed] and exposure)
  merged_oob <- rbindlist(current_oobs, use.names = TRUE, fill = TRUE)
  # Return the merged OOB data
  list(merged_oob = list(merged_oob))
}, by = seed]

# Convert the merged results to list format for easy access by seed
final_oob_list <- oob_merged$merged_oob
names(final_oob_list) <- oob_merged$seed

merged_oob_all <- Reduce(function(x, y) {
  # 使用merge函数按Exposure和Panel列合并
  merge(x, y, by = c("Exposure", "Panel"), all = TRUE)
}, final_oob_list)


list_disc_pro <- final_data_list
merged_oob_disc_pro <- merged_oob_all


#### merge pro and exp ####

merged_list_disc <- lapply(names(list_disc_pro), function(seed_name) {
  # 获取相同种子的蛋白质和表型数据框
  pro_df <- list_disc_pro[[seed_name]]
  exp_df <- list_disc_exp[[seed_name]]

  
  # 使用left_join按eid合并两个数据框
  merged_df <- left_join(pro_df, exp_df, by = "eid")
  
  return(merged_df)
})

# 为合并后的列表命名
names(merged_list_disc) <- names(list_disc_pro)

# 合并merged_oob_disc_pro和merged_oob_disc_exp
# 首先为merged_oob_disc_exp添加Panel列，设置为NA
merged_oob_disc_exp$Panel <- NA

# 按照相同的列名合并两个OOB数据框
merged_oob_all_disc <- bind_rows(
  merged_oob_disc_exp,
  merged_oob_disc_pro
)


save(merged_list_disc,merged_oob_all_disc,file = "/home/louchen/UKB_meno_pre/imputation/impute_data/disc_imp_surv_predi.RData")

print("save disc data")




#### dat_rep imputation ####


exp <- fread("/home/louchen/UKB_meno_pre/merge_and_extract/data_files/variable_define/exp_var_v1.txt")
out <- fread("/home/louchen/UKB_meno_pre/merge_and_extract/data_files/variable_define/out_surv_predi_v1.txt")
pro <- fread("/home/louchen/UKB_meno_pre/merge_and_extract/data_files/variable_define/pro_var_v0.txt")

##### clinical var #####

dat_rep_exp <- dat_rep %>%
  select(any_of(c("hazard", exp$Variable, out$Variable)))  


#exclude  some nested variables 

exclude_vars <- c(
  Cs(
    eid,
    time_follow
  )
)
exclude_pro <- c(
  Cs(
    
    fsh_ori,
    lh_ori,
    prl_ori
    
  )
)


dat_rep_exp <- dat_rep_exp %>% 
  select(-all_of(exclude_pro))


exclude_vars <- paste0(exclude_vars, collapse = " - ")
formula <- as.formula(paste0(". ~ . - ", exclude_vars))


# create case weights to weight down the contribution of rows with many missings during the imputation
non_miss_rep_exp <- rowSums(!is.na(dat_rep_exp))



rep_exp <- mclapply(3456:3460, function(x) {
  
  result <- missRanger(
    dat_rep_exp,
    formula = formula,
    maxiter = 10,
    pmm.k = 3,
    verbose = 1,
    seed = x,
    num.trees = 200,
    returnOOB = TRUE,
    case.weights = non_miss_rep_exp
  )
  
  oob <- as.data.frame(attr(result, "oob"))
  colnames(oob) <- c(paste0("Imputation_", x))
  oob$Exposure <- rownames(oob)
  
  seed_num<-paste0("Imputation_",x)
  
  return(list(
    seed = seed_num,
    result = result,
    oob = oob
  ))
  
  
}, mc.cores = 20)



results_dt  <- rbindlist(lapply(rep_exp, function(res) {
  data.table(
    seed = res$seed,
    result = list(res$result),
    oob = list(res$oob)
  )
}))



final_data_list <- results_dt$result
names(final_data_list) <- results_dt$seed


final_oob_list <- results_dt$oob
names(final_oob_list) <- results_dt$seed

merged_oob_all <- Reduce(function(x, y) {
  merge(x, y, by = c("Exposure"), all = TRUE)
}, final_oob_list)

list_rep_exp <- final_data_list
merged_oob_rep_exp <- merged_oob_all


##### protein #####

dat_rep_pro <- dat_rep %>%
  select(any_of(c("age_years",pro$Variable)))  

# import protein name with panel
pro_var_panel_distinct <- fread("/home/louchen/UKB_meno_pre/imputation/pro_var_panel_distinct.csv")

# get all Protein Panel
panels <- unique(pro_var_panel_distinct$Protein_panel)
print(panels)  # 8 different panel

# define seed 
seeds <- 3456:3460  # 5 seed for five different imputation dataset

# create panel-seed combination
tasks <- expand.grid(panel = panels,
                     seed = seeds,
                     stringsAsFactors = FALSE)


run_imputation <- function(task) {
  # get task panel and seed for each group intasks 
  current_panel <-   task$panel
  current_seed <-   task$seed
  
  # Extract the variables corresponding to the current panel
  panel_vars <- pro_var_panel_distinct[Protein_panel == current_panel, Variable]
  cols_to_keep <- c("eid","age_years", panel_vars)
  
  # Create a data subset (automatically skip nonexistent columns)
  dat_panel <- dat_rep_pro %>%
    select(any_of(cols_to_keep))
  
  #test code
  #dat_panel<-dat_panel[,1:4]
  
  
  # Generate formula (exclude EID), with only proteins
  formula <- as.formula(". ~ . - eid")
  
  # Calculate row missing weight
  non_miss <- rowSums(!is.na(dat_panel))
  
  # missranger
  result <- missRanger(
    dat_panel,
    formula = formula,
    maxiter = 10,
    pmm.k = 3,
    verbose = 1,
    seed = current_seed,
    num.trees = 200,
    returnOOB = TRUE,
    case.weights = non_miss
  )
  #Get the corresponding OOB
  oob <- as.data.frame(attr(result, "oob"))
  colnames(oob) <- c(paste0("Imputation_", current_seed))
  oob$Exposure <- rownames(oob)
  oob$Panel<- current_panel
  # Return results (panel name, seed, imputation data)
  result$age_years <- NULL
  
  seed_num<-paste0("Imputation_",current_seed)
  
  return(list(
    panel = current_panel,
    seed = seed_num,
    result = result,
    oob = oob
  ))
}



# Convert to list format for mclapply processing
tasks_list <- split(tasks, seq(nrow(tasks)))

# Perform parallel tasks
results <- mclapply(tasks_list, function(task) {
  run_imputation(task)
}, mc.cores = 20)

# rbind the lists from results(40 lists with panel-seed combination)
results_dt <- rbindlist(lapply(results, function(res) {
  data.table(
    panel = res$panel,
    seed = res$seed,
    result = list(res$result),
    oob = list(res$oob)
  )
}))




#merge the result in the result_dt,result column are List of a series of results
final_merged <- results_dt[, {
  if (.N == 8) {  # Ensure that each seed has 8 panels
    # Sort by panel order to ensure consistency of consolidation
    ordered_results <- result[order(match(panel, panels))]
    # Merge all data frames
    merged <- Reduce(function(x, y) full_join(x, y, by = "eid"), ordered_results)
    list(merged_data = list(merged))
  }
}, by = seed]

final_data_list <- final_merged$merged_data
names(final_data_list) <- final_merged$seed

oob_merged <- results_dt[, {
  # Get the OOB data of all panels under the current seed
  current_oobs <- oob
  # Use rbind to vertically merge all OOB data frames because they have the same column names (impression_[seed] and exposure)
  merged_oob <- rbindlist(current_oobs, use.names = TRUE, fill = TRUE)
  # Return the merged OOB data
  list(merged_oob = list(merged_oob))
}, by = seed]

# Convert the merged results to list format for easy access by seed
final_oob_list <- oob_merged$merged_oob
names(final_oob_list) <- oob_merged$seed

merged_oob_all <- Reduce(function(x, y) {
  # 使用merge函数按Exposure和Panel列合并
  merge(x, y, by = c("Exposure", "Panel"), all = TRUE)
}, final_oob_list)


list_rep_pro <- final_data_list
merged_oob_rep_pro <- merged_oob_all


#### merge pro and exp ####

merged_list_rep <- lapply(names(list_rep_pro), function(seed_name) {
  # 获取相同种子的蛋白质和表型数据框
  pro_df <- list_rep_pro[[seed_name]]
  exp_df <- list_rep_exp[[seed_name]]
  
  
  # 使用left_join按eid合并两个数据框
  merged_df <- left_join(pro_df, exp_df, by = "eid")
  
  return(merged_df)
})

# 为合并后的列表命名
names(merged_list_rep) <- names(list_rep_pro)

# 合并merged_oob_rep_pro和merged_oob_rep_exp
# 首先为merged_oob_rep_exp添加Panel列，设置为NA
merged_oob_rep_exp$Panel <- NA

# 按照相同的列名合并两个OOB数据框
merged_oob_all_rep <- bind_rows(
  merged_oob_rep_exp,
  merged_oob_rep_pro
)


save(merged_list_rep,merged_oob_all_rep,file = "/home/louchen/UKB_meno_pre/imputation/impute_data/rep_imp_surv_predi.RData")


print("save rep data")




















