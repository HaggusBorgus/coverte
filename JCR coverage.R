pacman::p_load("ggplot2", "forcats")

data <- read.csv("JCR coverage.csv")

data <- data |> dplyr::mutate(
  overall_low = rep(-1, 20),
  overall_high = rep(-1, 20),
  confidence_low = rep(-1, 20),
  confidence_high = rep(-1, 20),
  condition_low = rep(-1, 20),
  condition_high = rep(-1, 20)
)

for (i in 1:20){
  overall_cpi <- binom.test(x = data$Correct[i], 
                            n = data$N[i])$conf.int
  confidence_cpi <- binom.test(x = data$Interval.Correct[i], 
                               n = data$N[i])$conf.int
  condition_cpi <- binom.test(x = data$Correct[i], 
                              n = data$Interval.Correct[i])$conf.int
  data$overall_low[i] = overall_cpi[1]
  data$overall_high[i] = overall_cpi[2]
  data$confidence_low[i] = confidence_cpi[1]
  data$confidence_high[i] = confidence_cpi[2]
  data$condition_low[i] = condition_cpi[1]
  data$condition_high[i] = condition_cpi[2]
}

# Overall Coverage

plot <- ggplot(data |> 
  dplyr::mutate(Trial = paste(JCR.Method, Error.Type, sep = "/"),
                Method = fct_relevel(factor(Method), "ANOVA CATE", "IREG ATE", 
                                     "IREG CATE", "NDE", "CDE")),
  mapping = aes(x = Method,
                colour = Trial)) + 
  
  # Points
  
  geom_point(mapping = aes(y = (Correct/N))) +
  
  # Error Bars
  
  geom_errorbar(mapping = aes(ymin = overall_low, 
                              ymax = overall_high),
                width = 0.1) + 
  
  # 0.95 line
  
  geom_hline(yintercept = 0.95, linetype = "dashed", color = "black") +
  
  # Labels
  
  ggtitle("Overall Coverage") + labs(x = "Regression Specification") + 
  labs(y = "Probability") + 
  theme_bw()
plot

# Interval Coverage

plot <- ggplot(data |> 
  dplyr::mutate(Trial = paste(JCR.Method, Error.Type, sep = "/"),
                Method = fct_relevel(factor(Method), "ANOVA CATE", "IREG ATE", 
                                     "IREG CATE", "NDE", "CDE")),
  mapping = aes(x = Method,
                colour = Trial)) + 
  
  # Points
  
  geom_point(mapping = aes(y = (Interval.Correct/N))) +
  
  # Error Bars
  
  geom_errorbar(mapping = aes(ymin = confidence_low, 
                              ymax = confidence_high),
                width = 0.1) + 
  
  # 0.975 line
  
  geom_hline(yintercept = 0.975, linetype = "dashed", color = "black") +
  
  # Labels
  
  ggtitle("Interval Coverage") + labs(x = "Regression Specification") + 
  labs(y = "Probability") + 
  theme_bw()
plot

# Conditional Coverage

plot <- ggplot(data |> 
  dplyr::mutate(Trial = paste(JCR.Method, Error.Type, sep = "/"),
                Method = fct_relevel(factor(Method), "ANOVA CATE", "IREG ATE", 
                                     "IREG CATE", "NDE", "CDE")),
  mapping = aes(x = Method,
                colour = Trial)) + 
  
  # Points
  
  geom_point(mapping = aes(y = (Correct/Interval.Correct))) +
  
  # Error Bars
  
  geom_errorbar(mapping = aes(ymin = condition_low, 
                              ymax = condition_high),
                width = 0.1) + 
  
  # 0.975 line
  
  geom_hline(yintercept = 0.975, linetype = "dashed", color = "black") +
  
  # Labels
  
  ggtitle("Conditional Coverage") + labs(x = "Regression Specification") + 
  labs(y = "Probability") + 
  theme_bw()
plot
