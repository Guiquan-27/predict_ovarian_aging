
library(data.table)
library(dplyr)
panel<-fread("/home/louchen/UKB_meno_pre/imputation/pro_panel_ori.csv")
head(panel)
head(panel[,c("Assay Target","Protein panel")])
panel <- panel %>%
  select(`Assay Target`, `Protein panel`)


panel <- panel %>% 
  rename(Protein_panel = `Protein panel`)


# 查看原始唯一Protein_panel
unique(panel$Protein_panel)

#https://biobank.ndph.ox.ac.uk/showcase/refer.cgi?id=4654
# 按规则修正特定Assay Target的Protein_panel
panel <- panel %>%
  mutate(
   Protein_panel = case_when(
      `Assay Target` == "TNF"    ~ "Cardiometabolic",
      `Assay Target` == "IL6"    ~ "Oncology",
      `Assay Target` == "CXCL8"  ~ "Oncology",
      `Assay Target` == "IDO1"   ~ "Cardiometabolic_II",
      `Assay Target` == "LMOD1"  ~ "Neurology_II",
      `Assay Target` == "SCRIB"  ~ "Cardiometabolic_II",
      TRUE ~Protein_panel  # 其他情况保留原值
    )
  ) %>%
  # 去重：保证每个Assay Target对应唯一Protein_panel
  distinct(`Assay Target`,Protein_panel, .keep_all = FALSE)


pro_var<-fread("/home/louchen/UKB_meno_pre/imputation/pro_var_v1.txt")


pro_var_panel_distinct<-left_join(pro_var,panel,by = c("Protein" = "Assay Target"))

sum(is.na(pro_var_panel_distinct$Protein_panel))#0


fwrite(pro_var_panel_distinct,"/home/louchen/UKB_meno_pre/imputation/pro_var_panel_distinct.csv")


