# Generalized ROC AUC
To be selected for an internship, Yandex offered to solve the following problem.

Suggest we have $N$ objects. Calculate the generalized metric AUC, which determine by the following formula.
Where $t_i, t_j$ - target label and $y_i, y_j$ - prediction for object $i$ and $j$.

$$ \frac{\sum_{i,j: t_i > t_j} ([y_i > y_j] + \frac{1}{2}[y_i=y_j])}{| \{ i,j: t_i > t_j \} |}.$$






