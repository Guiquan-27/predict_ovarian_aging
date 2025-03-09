#set_ordinal_variables func



set_ordinal_variables <- function(data) {
  
  
  
  
  # smoke_status - Set as order factor
  data$smoke_status <- factor(
    data$smoke_status,
    ordered = TRUE,
    levels = c("Never", "Previous", "Current")
  )
  
  
  # age_first_birth_category - Set as order factor
  data$age_first_birth_category <- factor(
    data$age_first_birth_category,
    ordered = TRUE,
    levels = c("No_live_birth" ,"Age<=21", "Age22-25", "Age26-29", "Age>=30" )
  )
  
  # age_last_birth_category - Set as order factor
  data$age_last_birth_category <- factor(
    data$age_last_birth_category,
    ordered = TRUE,
    levels = c("No_live_birth","Age<=26", "Age27-30", "Age31-34", "Age>=35")
  )
  
  
  # length_mens_cyc_category - Set as order factor
  data$length_mens_cyc_category <- factor(
    data$length_mens_cyc_category,
    ordered = TRUE,
    levels = c(
       "Not_sure_other_reason", 
      "Irregular_cycle", "Day<=21", "Day22-27", "Day27-28", "Day>=29"
    )
  )
  # no "Menopause", "Not_sure_hysterectomy" ! in our dataset
  
  # oestradiol_category - Set as order factor
  data$oestradiol_category <- factor(
    data$oestradiol_category,
    ordered = TRUE,
    levels = c(
      "Lower_limit", "Oestradiol<=Q1", "Oestradiol>Q1<=Q2", 
      "Oestradiol>Q2<=Q3", "Oestradiol>Q3"
    )
  )
  # no Higher_limit in oestradiol_category  ! in our dataset
  
  # total_t_category - Set as order factor
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