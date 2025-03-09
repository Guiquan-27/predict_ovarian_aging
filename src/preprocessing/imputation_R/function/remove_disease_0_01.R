

remove_disease_0_01<- function (data_ori = data_ori){
  
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
  
  
  
  low_count_disease_vars <- c()
  
  
  for (var in disease_vars) {
    if (var %in% colnames(data_ori)) {
      yes_count <- sum(data_ori[[var]] == "Yes", na.rm = TRUE)
      threshold <- 0.01 * nrow(data_ori)
      # If the number of "yes" answers is less than the threshold, record the disease variable
      if (yes_count < threshold) {
        cat(
          sprintf(
            "Disease variable '%s' 'Yes'  answers are %d，Less than 1%% of the total sample  (threshold: %.1f)\n",
            var,
            yes_count,
            threshold
          )
        )
        low_count_disease_vars <- c(low_count_disease_vars, var)
      } else {
        cat(sprintf(
          "Disease variable'%s' 'Yes'  answers are  %d\n",
          var,
          yes_count
        ))
      }
    } else {
      cat(sprintf("Warning: disease variable '%s' is not in dataset\n", var))
    }
  }
  
  
  if (length(low_count_disease_vars) > 0) {
    cat("The following disease variables will be excluded（'Yes'Total number of answers 0.1%）:\n")
    for (var in low_count_disease_vars) {
      cat(sprintf("- %s\n", var))
    }
    
    # Exclude these disease variables from data_ori
    if (length(low_count_disease_vars) > 0) {
      data_ori <- data_ori %>% select(-all_of(low_count_disease_vars))
      cat(sprintf(
        "%d diseases and lesions have been excluded from data\n",
        length(low_count_disease_vars)
      ))
    }
  } else {
    cat("The 'yes' responses of all disease variables reached or exceeded the threshold, and no variables need to be excluded\n")
  }
  
  # 更新后的变量数
  cat(sprintf("Number of remaining variables in data_ori: %d\n", ncol(data_ori)))
  
  return(data_ori)

}


