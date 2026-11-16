import numpy as np
import matplotlib.pyplot as plt

# 0. קיבוע האקראיות לצורכי פיתוח ושחזור (Reproducibility)
np.random.seed(42)

# הגדרת ממדי הבעיה
n = 100  # מספר דוגמאות
m = 3    # מספר מאפיינים (פיצ'רים)

# 1. יצירת מטריצת הקלט X (התפלגות אחידה בין 0 ל-1)
X = np.random.rand(n, m)

# 2. קביעת הפרמטרים האמיתיים beta (חייב להיות וקטור עמודה m x 1)
beta = np.array([[2.5], 
                 [-4.0], 
                 [1.2]])

# 3. יצירת וקטור הרעש (התפלגות נורמלית סטנדרטית, sigma=1)
eps = np.random.randn(n, 1)

# 4. חישוב וקטור המטרה Y
Y = X @ beta + eps

# 5. פתרון המשוואות הנורמליות למציאת beta_hat
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ Y

# 6. השוואה בין האמת לאומדן
error = np.linalg.norm(beta - beta_hat)

print("True beta:\n", beta.flatten())
print("Estimated beta_hat:\n", np.round(beta_hat.flatten(), 4))
print(f"L2 Estimation Error: {error:.4f}")