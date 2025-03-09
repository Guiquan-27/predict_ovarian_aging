
imp_missranger <- function (dat_impu = dat_impu,
                            prefix = "impu",
                            save_folder = "/home/louchen/UKB_meno_pre/imputation/impute_data") {
  
  #test
  
  # dat_impu <- dat_disc
  # prefix = "disc"
  # save_folder = "/home/louchen/UKB_meno_pre/imputation/impute_data"
  
  #### load exposure and protein variable list ####
  
  #load exposure and protein variable list
  exp <- fread("/home/louchen/UKB_meno_pre/merge_and_extract/data_files/variable_define/exp_var_v1.txt")
  pro <- fread("/home/louchen/UKB_meno_pre/merge_and_extract/data_files/variable_define/pro_var_v0.txt")
  pro_var_panel_distinct <- fread("/home/louchen/UKB_meno_pre/imputation/pro_var_panel_distinct.csv")
  
  ##### clinical var #####
  
  dat_impu_exp <- dat_impu %>%
    select(any_of(c(exp$Variable)))  
  
  
  #exclude some nested variables 
  
  exclude_vars <- c(Cs(eid))
  
  exclude_pro <- c(Cs(fsh_ori, lh_ori, prl_ori))
  
  
  dat_impu_exp <- dat_impu_exp %>%
    select(-all_of(exclude_pro))
  
  
  exclude_vars <- paste0(exclude_vars, collapse = " - ")
  formula <- as.formula(paste0(". ~ . - ", exclude_vars))
  
  
  # create case weights to weight down the contribution of rows with many missings during the imputation
  non_miss_impu_exp <- rowSums(!is.na(dat_impu_exp))
  
  
  
  impu_exp <- mclapply(3456:3460, function(x) {
    
    
    seed_num<-paste0("Imputation_",x)
    
    print(paste0("now is running ",seed_num))
    
    
    result <- missRanger(
      dat_impu_exp,
      formula = formula,
      maxiter = 10,
      pmm.k = 3,
      verbose = 0,
      seed = x,
      num.trees = 200,
      returnOOB = TRUE,
      case.weights = non_miss_impu_exp
    )
    
    oob <- as.data.frame(attr(result, "oob"))
    colnames(oob) <- c(paste0("Imputation_", x))
    oob$Exposure <- rownames(oob)
    
    print(paste0("running over ",seed_num))
    
    return(list(
      seed = seed_num,
      result = result,
      oob = oob
    ))
    
    
  }, mc.cores = 20)
  
  
  
  results_dt  <- rbindlist(lapply(impu_exp, function(res) {
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
  
  list_impu_exp <- final_data_list
  merged_oob_impu_exp <- merged_oob_all
  
  
  ##### protein #####
  
  dat_impu_pro <- dat_impu %>%
    select(any_of(c("age_years",pro$Variable)))  
  
  
  
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
    dat_panel <- dat_impu_pro %>%
      select(any_of(cols_to_keep))
    
    #test code
    dat_panel<-dat_panel[,1:4]
    
    
    # Generate formula (exclude EID), with only proteins
    formula <- as.formula(". ~ . - eid")
    
    # Calculate row missing weight
    non_miss <- rowSums(!is.na(dat_panel))
    
    
    
    seed_num<-paste0("Imputation_",current_seed)
    
    print(paste0("now is running ", seed_num, "  ", current_panel))
    
    
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
    
    print(paste0("running over ", seed_num, "  ", current_panel))
    
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
  }, mc.cores = 30)
  
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
  
  
  list_impu_pro <- final_data_list
  merged_oob_impu_pro <- merged_oob_all
  
  
  ##### merge pro and clinical ####
  
  merged_list_impu <- lapply(names(list_impu_pro), function(seed_name) {
    # 获取相同种子的蛋白质和表型数据框
    pro_df <- list_impu_pro[[seed_name]]
    exp_df <- list_impu_exp[[seed_name]]
    
    
    # 使用left_join按eid合并两个数据框
    merged_df <- left_join(pro_df, exp_df, by = "eid")
    
    return(merged_df)
  })
  
  # 为合并后的列表命名
  names(merged_list_impu) <- names(list_impu_pro)
  
  # 合并merged_oob_impu_pro和merged_oob_impu_exp
  # 首先为merged_oob_impu_exp添加Panel列，设置为NA
  merged_oob_impu_exp$Panel <- NA
  
  # 按照相同的列名合并两个OOB数据框
  merged_oob_all_impu <- bind_rows(
    merged_oob_impu_exp,
    merged_oob_impu_pro
  )
  
  
  save(merged_list_impu,merged_oob_all_impu, file = 
         paste0(save_folder,"/",prefix,"_imp.RData"))
  
  cat(prefix,"_imp run over \nsave into \n", save_folder, "/", prefix, "_imp.RData\n", sep = "")


}
