# Yandex-task
To be selected for an internship, Yandex offered to solve the following problem.

*Calculate the generalized metric AUC, which determine by the following formula:* 

$$ \frac{\sum_{i,j: t_i > t_j} ([y_i > y_j] + \frac{1}{2}[y_i=y_j])}{| \{ i,j: t_i > t_j \} |},$$

*where $t_i, t_j$ - target label and $y_i, y_j$ - prediction for object $i$ and $j$.*
