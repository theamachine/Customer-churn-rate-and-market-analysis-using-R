library(arules)
library(RColorBrewer)
dataset <- read.transactions("transactions.csv", format = 'basket', sep = ',', )
dataset


itemFrequencyPlot(items(dataset), topN = 10,
                  col = brewer.pal(8, 'Pastel2'),
                  main = 'top 10 transactions',
                  type = "relative",
                  ylab = "Item Frequency (Relative)"
                  )

summary(dataset)

arule_l3 <- sort (apriori(dataset, parameter = list(maxlen=3,support = 0.002, confidence = 0.20)), by="lift")
inspect(arule_l3)

arule_l2 <- sort (apriori(dataset, parameter = list(maxlen=2,support = 0.002, confidence = 0.20)), by="lift")
inspect(arule_l2)

inspect(arule_l3[1])

inspect(arule_l2[1])



