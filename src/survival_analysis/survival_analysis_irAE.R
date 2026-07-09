# Survival analysis for irAE timeline project

library(ggplot2)
library(dplyr)
library(survival)
library(survminer)

# set working directory to root of proj
setwd("Documents/Tan_Lab/Projects/irAE_timeline_plots_LLM/irAE_timeline_plots_LLM/")

# read in file with survival data
surv_table = read.csv("data/survival_data_labeled.csv", header = T)

# Summarize # of irAE vs non irAE px idenfied
table(surv_table$irAE)

# Summarize # of px with each oncotree tissue
table(surv_table$oncotree_tissue)

# Summarize # of px with each oncotree tissue and irAE status
table(surv_table$oncotree_tissue, surv_table$irAE)

# KM curve irAE vs no irAE - overall cohort only after ICI
survival_fit = survfit(Surv(rwOS_ICI_months,Is.Deceased.) ~ irAE, data = surv_table)
ggsurvplot(survival_fit, conf.int=T, pval = T, pval.method = T, risk.table = T, legend.title = "irAE Status",
           legend.labs = c("irAE","No irAE"), surv.median.line="hv", xlab = "Time (Months)",
           palette = c('maroon','darkcyan'))
# Cox PH model 
surv_table$irAE <- relevel(
  factor(surv_table$irAE),
  ref = "no irAE"
)
cox = coxph(Surv(rwOS_ICI_months,Is.Deceased.) ~ irAE, surv_table) 
summary(cox)

# KM curve irAE vs no irAE - only skin cancer px after ICI
mel_table = surv_table %>% filter(oncotree_tissue == "Skin")
survival_fit = survfit(Surv(rwOS_ICI_months,Is.Deceased.) ~ irAE, data = mel_table)
ggsurvplot(survival_fit, conf.int=T, pval = T, pval.method = T, risk.table = T, legend.title = "irAE Status",
           legend.labs = c("irAE","No irAE"), surv.median.line="hv",xlab = "Time (Months)",
           palette = c('maroon','darkcyan'))

# KM curve irAE vs no irAE - only lung cancer px after ICI
lung_table = surv_table %>% filter(oncotree_tissue == "Lung")
survival_fit = survfit(Surv(rwOS_ICI_months,Is.Deceased.) ~ irAE, data = lung_table)
ggsurvplot(survival_fit, conf.int=T, pval = T, pval.method = T, risk.table = T, legend.title = "irAE Status",
           legend.labs = c("irAE","No irAE"), surv.median.line = "hv",xlab = "Time (Months)",
           palette = c('maroon','darkcyan'))

# KM curve within irAE samples - cutaneous/endocrine vs lung/cardiovascular
irae_table = surv_table %>% filter(irAE == "irAE")
table(irae_table$irAE)
irae_table = irae_table %>% dplyr::mutate(irae_type_consolidated = dplyr::case_when(irae_type_Lung == 1 | irae_type_Cardiovascular == 1 ~ "Lung_Cardiovascular",
                                                                                    irae_type_Cutaneous == 1 | irae_type_Endocrine == 1 ~ "Cutaneous_Endocrine"))
table(irae_table$irae_type_consolidated)
table(irae_table$irae_type_Lung)
table(irae_table$irae_type_Cardiovascular)

survival_fit = survfit(Surv(rwOS_ICI_months,Is.Deceased.) ~ irae_type_consolidated, data = irae_table)
ggsurvplot(survival_fit, conf.int=T, pval = T, pval.method = T, risk.table = T, legend.title = "irAE Status",
           legend.labs = c("Cutaneous/Endocrine","Lung/Cardiovascular"), surv.median.line = "hv",xlab = "Time (Months)",
           palette = c('maroon','darkcyan'))
# Cox PH model 
irae_table$irae_type_consolidated <- relevel(
  factor(irae_table$irae_type_consolidated),
  ref = "Lung_Cardiovascular"
)
cox = coxph(Surv(rwOS_ICI_months,Is.Deceased.) ~ irae_type_consolidated, irae_table) 
summary(cox)

# KM curve within irAE samples = all four categories
# order of priority for keeping samples with both : cardiovascular > lung > endocrine > cutaneous
irae_table = irae_table %>% dplyr::mutate(irae_type_separate = dplyr::case_when(irae_type_Cardiovascular == 1 ~ "Cardiovascular",
                                                                                irae_type_Lung == 1 ~ "Lung",
                                                                                irae_type_Endocrine == 1 ~ "Endocrine",
                                                                                irae_type_Cutaneous == 1 ~ "Cutaneous"))
table(irae_table$irae_type_separate)
table(irae_table$irae_type_Lung)
table(irae_table$irae_type_Cardiovascular)

survival_fit = survfit(Surv(rwOS_ICI_months,Is.Deceased.) ~ irae_type_separate, data = irae_table)
ggsurvplot(survival_fit, conf.int=F, pval = T, pval.method = T, risk.table = T, legend.title = "irAE Status",
           legend.labs = c("Cardiovascular","Cutaneous","Endocrine","Lung"), surv.median.line = "hv",xlab = "Time (Months)",
           palette = c('maroon','pink2','darkcyan','lightblue3')
           )
           



