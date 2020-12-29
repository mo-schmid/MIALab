library(carData)
library(car)
library(readxl)
library(readr)
library(lattice)
library(BSDA)



# select result folder
setwd("C:/Users/morit/OneDrive - Universitaet Bern/02_Master UniBe/03_Semester/01_Medical Image Analysis Lab/04_project/MIALab/bin/mia-result/")



# import the data for PKF
ref <- data.frame(read_delim("gridsearch_PKF/2020-12-11-09-51-54/no_PP/results.csv",
                  ";", escape_double = FALSE, trim_ws = TRUE))

ref$SUBJECT <- as.character(ref$SUBJECT)

pp <- data.frame(read_delim("gridsearch_PKF/2020-12-11-09-51-54/with_PP/PP-V-20_0-BG-True/results.csv",
                            ";", escape_double = FALSE, trim_ws = TRUE))


# # import data for CRF
# ref <- data.frame(read_delim("gridsearch_CRF/no_PP_best/results.csv",
#                              ";", escape_double = FALSE, trim_ws = TRUE))
# 
# ref$SUBJECT <- as.character(ref$SUBJECT)
# 
# pp <- data.frame(read_delim("gridsearch_CRF/with_PP_best/results.csv",
#                             ";", escape_double = FALSE, trim_ws = TRUE))



# create data frame to store the statistical results
statResults <- data.frame(LABEL = character(),
                          METRIC = character(),
                          STAT_VAL = character(),
                          NO_PP = double(),
                          WITH_PP = double(),
                          DIFF = double(),
                          P_VAL = double(),
                          SIGNIFICANT = numeric(),
                          CI95_LOW = double(),
                          CI95_HIGH = double())

for (l in unique(ref$LABEL))
{
  # t-test
  res <- t.test(subset(pp, LABEL == l)$DICE,subset(ref, LABEL == l)$DICE ,paired = TRUE, alternative = "two.sided" )

  tmp <- data.frame(LABEL = l,
                    METRIC = "DICE",
                    STAT_VAL = "Mean",
                    NO_PP = mean(subset(ref, LABEL == l)$DICE),
                    WITH_PP = mean(subset(pp, LABEL == l)$DICE),
                    DIFF = unname(res$estimate),
                    P_VAL = res$p.value,
                    SIGNIFICANT = (res$p.value < 0.05),
                    CI95_LOW = res$conf.int[1],
                    CI95_HIGH = res$conf.int[2])

  statResults <- rbind(statResults, tmp)
  
  # Sign Test
  res <- SIGN.test(subset(pp, LABEL == l)$HDRFDST,subset(ref, LABEL == l)$HDRFDST ,paired = TRUE, alternative = "two.sided" )
  
  tmp <- data.frame(LABEL = l,
                    METRIC = "HDRFDST",
                    STAT_VAL = "Median",
                    NO_PP = median(subset(ref, LABEL == l)$HDRFDST),
                    WITH_PP = median(subset(pp, LABEL == l)$HDRFDST),
                    DIFF = unname(res$estimate),
                    P_VAL = res$p.value,
                    SIGNIFICANT = (res$p.value < 0.05),
                    CI95_LOW = res$conf.int[1],
                    CI95_HIGH = res$conf.int[2])

  statResults <- rbind(statResults, tmp)
  
}


# save results as csv

write.csv(statResults,"stat-result/stat_pkf.csv", row.names = TRUE )

# write.csv(statResults,"stat-result/stat_CRF.csv", row.names = TRUE )



